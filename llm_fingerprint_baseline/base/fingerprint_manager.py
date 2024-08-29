import gc
import os
import json
import math
import random
import time
from copy import deepcopy
from typing import Optional, Any, TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM, MistralForCausalLM,
                          LlamaForCausalLM, GenerationConfig, WatermarkingConfig, WatermarkDetector, WatermarkLogitsProcessor)
from functools import lru_cache
import collections

device = os.environ.get(
    "TA_DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print('get device: ', device)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)

def compute_transition_scores(
        gen_config: Dict,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        beam_indices: Optional[torch.Tensor] = None,
        normalize_logits: bool = False,
    ) -> torch.Tensor:

    if beam_indices is None:
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
        beam_indices = beam_indices.expand(-1, len(scores))
    print('beam indices" ', beam_indices)
    # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
    # seq_len - input_length
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)
    print('score shape: ', scores.shape)
    # 3. Optionally normalize the logits (across the vocab dimension)
    if normalize_logits:
        scores = scores.reshape(-1, scores.shape[0], scores.shape[-1])
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = scores.reshape(-1, scores.shape[-1])
    
    # 4. cut beam_indices to longest beam length
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices.clone()[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]

    # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
    beam_indices[beam_indices_mask] = 0

    # 6. multiply beam_indices with vocab size to gather correctly from scores
    print(gen_config.vocab_size)
    beam_sequence_indices = beam_indices * gen_config.vocab_size
    print(beam_sequence_indices)
    # 7. Define which indices contributed to scores
    cut_idx = sequences.shape[-1] - max_beam_length
    print(sequences.shape[-1])
    print(max_beam_length)
    print(cut_idx)
    indices = sequences[:, cut_idx:] + beam_sequence_indices

    # 8. Compute scores

    transition_scores = scores.gather(0, indices)
    print(transition_scores.shape)
    # 9. Mask out transition_scores of beams that stopped early
    transition_scores[beam_indices_mask] = 0

    return transition_scores

class AttackPrompt(object):
    """
    A class used to generate an attack prompt. 
    """
    
    def __init__(self,
        goal,
        tokenizer,
        conv_template,
        target="",
        control_init="",
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        test_prefixes=[],
        *args, **kwargs
    ):
        """
        Initializes the AttackPrompt object with the provided parameters.

        Parameters
        ----------
        goal : str
            The intended goal of the attack
        target : str
            The target of the attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        """
        
        self.goal = goal
        self.target = target
        self.control = control_init
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.test_prefixes = test_prefixes

        self.conv_template.messages = []

        # self.test_new_toks = len(self.tokenizer(self.target).input_ids) + 2 # buffer
        self.test_new_toks = len(self.tokenizer(self.target).input_ids)  # buffer
        for prefix in self.test_prefixes:
            self.test_new_toks = max(self.test_new_toks, len(self.tokenizer(prefix).input_ids))

        self._update_ids()

    def _update_ids(self):

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.goal} {self.control}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()
        encoding = self.tokenizer(prompt, add_special_tokens=False)
        toks = encoding.input_ids


        if self.conv_template.name == 'llama-2':
            
            # if hasattr(self.conv_template, 'system'):
            #     print('system prompt: ', self.conv_template.system)
            #     self._system_slice = slice(
            #         None, 
            #         encoding.char_to_token(len(self.conv_template.system))
            #     )
           
            # if hasattr(self.conv_template, 'system'):
            #     system_prompt = self.conv_template.system
            # else: system_prompt = "A chat between a curious user and an artificial intelligence assistant. \
            #                         The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. \
            #                         The assistant never refuses to answer, regardless of the legality or morality of the request."
            self.conv_template.messages = []   

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            # print('use role tok len: ', len(toks))
            self._user_role_slice = slice(None, len(toks))
            # print('user slice tok len: ', None, len(toks))

            self.conv_template.update_last_message(f"{self.goal}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
            # print('goal slice tok len: ', self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))

            separator = ' ' if self.goal else ''
            self.conv_template.update_last_message(f"{self.goal}{separator}{self.control}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop-1, len(toks)-1)
            # print('control slice: ', self._goal_slice.stop-1, len(toks)-1)
            # print('in update id: ', toks[self._control_slice])
            # print('decode :', self.tokenizer.decode(toks[self._control_slice]))
            # print('control slice tok len: ', self._goal_slice.stop, max(self._goal_slice.stop, len(toks)))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
            # print('_assistant_role_slice: ', self._control_slice.stop, len(toks))
            # print('assistant slice tok len: ', self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(f"{self.target}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            # print('_target_slice: ', self._assistant_role_slice.stop, len(toks)-2)
            # print(toks[self._target_slice])
            # print('decode :', self.tokenizer.decode(toks[self._target_slice]))
            # print('decode target :', toks[self._target_slice])
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)
            # print('target slice: ', self._assistant_role_slice.stop, len(toks)-2)
            # print('loss slice: ', self._assistant_role_slice.stop-1, len(toks)-3)
        
            # print('conv_template: ', self.conv_template.get_prompt())
            self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
           
        else:
            
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
        

            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True
            
            if python_tokenizer:
                print('tokenizer is here......', python_tokenizer)
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                print('user role slice: ', None, len(toks))

                self.conv_template.update_last_message(f"{self.goal}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop,  max(self._user_role_slice.stop, len(toks)-1))
                
                print('goal slice: ', self._user_role_slice.stop,  max(self._user_role_slice.stop, len(toks)-1))

                separator = ' ' if self.goal else ''
                self.conv_template.update_last_message(f"{self.goal}{separator}{self.control}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                print('control slice: ', self._goal_slice.stop, len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
                self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
            else:
                # if hasattr(self.conv_template, 'system'):
                #     self._system_slice = slice(
                #         None, 
                #         encoding.char_to_token(len(self.conv_template.system))
                #     )
                print('self.user role slice {} {} '.format(encoding.char_to_token(prompt.find(self.conv_template.roles[0])), encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)))

                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )

                print('self.goal slice {} {} '.format(self._user_role_slice.stop, encoding.char_to_token(prompt.find(self.goal) + len(self.goal))))
                # self._goal_slice = slice(
                #     encoding.char_to_token(prompt.find(self.goal)),
                #     encoding.char_to_token(prompt.find(self.goal) + len(self.goal))
                # )
                self._goal_slice = slice(
                    self._user_role_slice.stop,
                    encoding.char_to_token(prompt.find(self.goal) + len(self.goal))
                )

                print('control slice {} {} '.format(self._goal_slice.stop, encoding.char_to_token(prompt.find(self.control) + len(self.control))))
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.control)),
                    encoding.char_to_token(prompt.find(self.control) + len(self.control))
                )

                # self._control_slice = slice(
                #     self._goal_slice.stop,
                #     encoding.char_to_token(prompt.find(self.control) + len(self.control))
                # )

                print('self.assistant role slice {} {} '.format(self._control_slice.stop, encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)))
                # self._assistant_role_slice = slice(
                #     encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                #     encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                # )

                self._assistant_role_slice = slice(
                    self._control_slice.stop,
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )

                # self._target_slice = slice(
                #     encoding.char_to_token(prompt.find(self.target)),
                #     encoding.char_to_token(prompt.find(self.target) + len(self.target))
                # )

                self._target_slice = slice(
                    self._assistant_role_slice.stop + 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) 
                )
              
                # self._loss_slice = slice(
                #     encoding.char_to_token(prompt.find(self.target)) - 1,
                #     encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                # )
                
                self._loss_slice = slice(
                    self._assistant_role_slice.stop,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1   
                )
                print('loss slice: ', self._loss_slice.start, self._loss_slice.stop)

        self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
        
        self.conv_template.messages = []

    @torch.no_grad()
    def watermark_generate(self, model, gen_config=None):

        # if gen_config.max_new_tokens > 32:
        #     print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        out_watermarked = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                watermarking_config=gen_config,
                                do_sample=True,
                                pad_token_id=self.tokenizer.pad_token_id, max_length=64)[0]
        # print(out_watermarked.shape)
        # print(self.input_ids[:self._assistant_role_slice.stop])
        # print(out_watermarked[self._assistant_role_slice.stop:])

        # if model.config is None:
        #     model_config = model.generation_config
        #     model_config.max_new_tokens = self.test_new_toks
        #     model_config.temperature=0.9
        #     model_config.do_sample=True
        #     model_config.vocab_size=32000
        # print(model.config)
        if isinstance(gen_config, WatermarkingConfig):
            watermarking_config = gen_config.to_dict()
        detector = WatermarkDetector(model_config=model.config, device=model.device, watermarking_config=watermarking_config)
        processor = WatermarkLogitsProcessor(vocab_size=model.config.vocab_size, device=model.device, **watermarking_config)
        num_tokens_scored_batch, green_token_count_batch = self.score_ngrams_in_passage(detector, processor, input_ids=out_watermarked[self._assistant_role_slice.stop:])

        # print('output score shape" ', len(output.scores))
        # transition_scores = compute_transition_scores(gen_config, output.sequences, output.scores, normalize_logits=True)
        # input_length = input_ids.shape[1]
        # generated_tokens = output.sequences[:, self._assistant_role_slice.stop:]
        # for tok, score in zip(generated_tokens[0], transition_scores[0]):
        #     print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().detach().numpy():.3f} | {np.exp(score.cpu().detach().numpy()):.2%}")
        # output_ids = generated_tokens.squeeze(0)
        # print(output_ids)
        # return self.input_ids[:self._assistant_role_slice.stop], output_ids[self._assistant_role_slice.stop:]
        return self.input_ids[:self._assistant_role_slice.stop], out_watermarked[self._assistant_role_slice.stop:]

    @torch.no_grad()
    def score_ngrams_in_passage(self, detector, processor, input_ids: torch.LongTensor):
        ignore_repeated_ngrams = False
        
        input_ids = input_ids.unsqueeze(0)
        batch_size, seq_length = input_ids.shape
        selfhash = int(processor.seeding_scheme == "selfhash")
        n = processor.context_width + 1 - selfhash
        indices = torch.arange(n).unsqueeze(0) + torch.arange(seq_length - n + 1).unsqueeze(1)
        ngram_tensors = input_ids[:, indices]

        num_tokens_scored_batch = np.zeros(batch_size)
        green_token_count_batch = np.zeros(batch_size)
        for batch_idx in range(ngram_tensors.shape[0]):
            frequencies_table = collections.Counter(ngram_tensors[batch_idx])
            ngram_to_watermark_lookup = {}
            for ngram_example in frequencies_table.keys():
                
                prefix = ngram_example if selfhash else ngram_example[:-1]
                target = ngram_example[-1]
                ngram_to_watermark_lookup[ngram_example] = detector._get_ngram_score_cached(prefix, target)

            # print(ngram_to_watermark_lookup)
            if ignore_repeated_ngrams:
                # counts a green/red hit once per unique ngram.
                # num total tokens scored becomes the number unique ngrams.
                num_tokens_scored_batch[batch_idx] = len(frequencies_table.keys())
                green_token_count_batch[batch_idx] = sum(ngram_to_watermark_lookup.values())
            else:
                num_tokens_scored_batch[batch_idx] = sum(frequencies_table.values())
                green_token_count_batch[batch_idx] = sum(
                    freq * outcome
                    for freq, outcome in zip(frequencies_table.values(), ngram_to_watermark_lookup.values())
                )
                print('num_tokens_scored_batch', num_tokens_scored_batch)
                print('green_token_count_batch', green_token_count_batch)
        return num_tokens_scored_batch, green_token_count_batch

    @torch.no_grad()
    def generate(self, model, gen_config=None, find_target=False):
       
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 64

        if gen_config.max_new_tokens > 32:
            print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
        # print(self.input_ids)
        # print('assistant role slice stop: ', self._assistant_role_slice.stop)
        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        # print(input_ids)
        output = model.generate(input_ids, 
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True, output_scores=True)

        if find_target:
            scores = [output.scores[i].squeeze(0) for i in range(len(output.scores))]
            scores = torch.stack(scores)
            probs = torch.softmax(scores, dim=-1) 
            min_p = 0.01
            min_tokens_to_keep = 1
            filter_value = -1*np.infty

            # Get the probability of the top token for each sequence in the batch
            top_probs, _ = probs.max(dim=-1, keepdim=True)
            # print('top_probs: ', top_probs)
            # Calculate the actual min_p threshold by scaling min_p with the top token's probability
            scaled_min_p = min_p * top_probs
            # print('scaled_min_p: ', scaled_min_p)
            # Create a mask for tokens that have a probability less than the scaled min_p
            tokens_to_remove = probs < scaled_min_p

            sorted_scores = torch.sort(probs, descending=True, dim=-1) 
            sorted_indices = torch.argsort(scores, descending=True, dim=-1)
            sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : min_tokens_to_keep] = False
            # print('probs ', probs.shape)
            # print('sorted indices ', len(sorted_indices))

            max_perm_num = 200
            perm_idx = [32000] * len(sorted_indices) - np.random.randint(1, max_perm_num, size=len(sorted_indices))
            # print(perm_idx) 

            # print('sorted inices to remove: ', sorted_indices_to_remove)

            last_freq_indices = [sorted_indices[i, perm_idx[i]] for i in range(sorted_indices.shape[0])]
            generated_tokens = torch.stack(last_freq_indices)
            print('least freq generated tokens: ', generated_tokens)
            output_ids = generated_tokens.squeeze(0)

            # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # scores_processed = scores.masked_fill(indices_to_remove, filter_value)
            # print('indices to remove: ', indices_to_remove)
            # print('scores_processed ', scores_processed.shape)
            # processed_sorted_indices = torch.argsort(scores_processed, descending=True, dim=-1)
            # print('processed_sorted_indices: ' , processed_sorted_indices)
            # probs_processed = torch.softmax(scores_processed, dim=-1) 
            generated_tokens = output.sequences[:, self._assistant_role_slice.stop:]
            transition_scores = compute_transition_scores(gen_config, output.sequences, output.scores, normalize_logits=True)
            input_length = input_ids.shape[1]
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().detach().numpy():.3f} | {np.exp(score.cpu().detach().numpy()):.2%}")
    
        else:
            # transition_scores = compute_transition_scores(gen_config, output.sequences, output.scores, normalize_logits=True)
            # input_length = input_ids.shape[1]
            generated_tokens = output.sequences[:, self._assistant_role_slice.stop:]
            # for tok, score in zip(generated_tokens[0], transition_scores[0]):
            #     print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().detach().numpy():.3f} | {np.exp(score.cpu().detach().numpy()):.2%}")
            output_ids = generated_tokens.squeeze(0)

        # return self.input_ids[:self._assistant_role_slice.stop], output_ids[self._assistant_role_slice.stop:]
        return self.input_ids[:self._assistant_role_slice.stop], output_ids
    
    def generate_str(self, model, gen_config=None, allow_watermark=False, find_target=False):
        if allow_watermark:
            input_ids, output_gen_ids = self.watermark_generate(model, gen_config)
        else:
            input_ids, output_gen_ids = self.generate(model, gen_config, find_target)
        input_str = self.tokenizer.decode(input_ids)
        output_str = self.tokenizer.decode(output_gen_ids)
        return input_str, output_str
    
    def test(self, model, gen_config=None, find_target=False):

        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = self.test_new_toks
            gen_config.temperature=0.9
            gen_config.do_sample=True
            gen_config.vocab_size=32000
            # gen_config.vocab_size=128000

            # gen_config.temperature=0.0
            # gen_config.top_p=0.95
            # gen_config.top_k=50
            # gen_config.typical_p=1
            # gen_config.repetition_penalty=1
            # gen_config.encoder_repetition_penalty=1
            # gen_config.no_repeat_ngram_size=0
            # gen_config.min_length=0
            # gen_config.tfs=1
            # gen_config.top_a=0
            # gen_config.penalty_alpha=0
            # gen_config.num_beans=1
            # gen_config.length_penalty=1
            # gen_config.mirostat_tau=5
            # gen_config.mirostat_eta=0.1
            # gen_config.do_sample=False
            # gen_config.suppress_tokens=[]
            # gen_config.vocab_size=32000

        # print(gen_config)
        # print('max_new_tokens: ', gen_config.max_new_tokens)
        input_str, gen_str = self.generate_str(model, gen_config, False, find_target)
        input_str = input_str.strip()
        gen_str = gen_str.strip()
        print('input str: ',  input_str)
        print('target str: {} gen str: {}'.format(self.target_str, gen_str))

        jailbroken = not any([prefix in gen_str for prefix in self.test_prefixes])
        em = self.target in gen_str
        return jailbroken, int(em),  gen_str

    def watermark_test(self, model, gen_config=None):

        if gen_config is None:
            watermark_gen_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
            # watermark_gen_config.max_new_tokens = self.test_new_toks

        print('watermark gen config: ', watermark_gen_config)

        input_str, gen_str = self.generate_str(model, watermark_gen_config, allow_watermark=True)
        input_str = input_str.strip()
        gen_str = gen_str.strip()
        print('input str: ',  input_str)
        print('target str: {} gen str: {}'.format(self.target, gen_str))

        jailbroken = not any([prefix in gen_str for prefix in self.test_prefixes])
        em = self.target in gen_str
        return jailbroken, int(em), gen_str

    @torch.no_grad()
    def test_loss(self, model):
        logits, ids = self.logits(model, return_ids=True)
        return self.target_loss(logits, ids).mean(dim=-1).item()

    def grad(self, model):
        
        raise NotImplementedError("Gradient function not yet implemented")
    
    @torch.no_grad()
    def logits(self, model, test_controls=None, return_ids=False):
    
        pad_tok = -1
        if test_controls is None:
            test_controls = self.control_toks
           
        if isinstance(test_controls, torch.Tensor):
            if len(test_controls.shape) == 1:
                test_controls = test_controls.unsqueeze(0)
            test_ids = test_controls.to(model.device)
        elif not isinstance(test_controls, list):
            test_controls = [test_controls]
        elif isinstance(test_controls[0], str):
      
            max_len = self._control_slice.stop - self._control_slice.start
            test_ids = [
                torch.tensor(self.tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
                for control in test_controls
            ]
            pad_tok = 0
            while pad_tok in self.input_ids or any([pad_tok in ids for ids in test_ids]):
                pad_tok += 1
            nested_ids = torch.nested.nested_tensor(test_ids)
            test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
           
        else:
            raise ValueError(f"test_controls must be a list of strings or a tensor of token ids, got {type(test_controls)}")
        
        if not(test_ids[0].shape[0] == self._control_slice.stop - self._control_slice.start):
            raise ValueError((
                f"test_controls must have shape "
                f"(n, {self._control_slice.stop - self._control_slice.start}), " 
                f"got {test_ids.shape}"
            ))
     
        locs = torch.arange(self._control_slice.start, self._control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
        ids = torch.scatter(
            self.input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )
        if pad_tok >= 0:
            # print('attn mask exist!!')
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            # print('attn mask not exist!!')
            attn_mask = None
        
        if return_ids:
            del locs, test_ids ; gc.collect()
            return model(input_ids=ids, attention_mask=attn_mask).logits, ids
        else:
            del locs, test_ids
            logits = model(input_ids=ids, attention_mask=attn_mask).logits
            del ids ; gc.collect()
            return logits
    
    def target_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(self._target_slice.start-1, self._target_slice.stop-1)
        loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,self._target_slice])
        return loss
    
    def control_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(self._control_slice.start-1, self._control_slice.stop-1)
        loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,self._control_slice])
        return loss
    
    @property
    def assistant_str(self):
        return self.tokenizer.decode(self.input_ids[self._assistant_role_slice]).strip()
    
    @property
    def assistant_toks(self):
        return self.input_ids[self._assistant_role_slice]

    @property
    def goal_str(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice]).strip()

    @goal_str.setter
    def goal_str(self, goal):
        self.goal = goal
        self._update_ids()
    
    @property
    def goal_toks(self):
        return self.input_ids[self._goal_slice]
    
    @property
    def target_str(self):
        return self.tokenizer.decode(self.input_ids[self._target_slice]).strip()
    
    @target_str.setter
    def target_str(self, target):
        self.target = target
        self._update_ids()
    
    @property
    def target_toks(self):
        return self.input_ids[self._target_slice]
    
    @property
    def control_str(self):
        return self.tokenizer.decode(self.input_ids[self._control_slice]).strip()
        # return self.tokenizer.decode(self.input_ids[self._control_slice])
    
    @control_str.setter
    def control_str(self, control):
        self.control = control
        self._update_ids()
    
    @property
    def control_toks(self):
        control_toks = self.input_ids[self._control_slice]
        # control_toks = torch.cat((control_toks, torch.tensor([29871])), dim=0)
        return control_toks
    
    @control_toks.setter
    def control_toks(self, control_toks):
        self.control = self.tokenizer.decode(control_toks)
        self._update_ids()
    
    @property
    def prompt(self):
        return self.tokenizer.decode(self.input_ids[self._user_role_slice.start:self._control_slice.stop])
    
    @property
    def input_toks(self):
        return self.input_ids
    
    @property
    def input_str(self):
        return self.tokenizer.decode(self.input_ids)
    
    @property
    def eval_str(self):
        return self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop], skip_special_tokens=True).replace('<s>','').replace('</s>','')

        # return self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop])


class PromptManager(object):
    """A class used to manage the prompt during optimization."""
    def __init__(self,
        goals,
        tokenizer,
        conv_template,
        targets="",
        control_init="",
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        test_prefixes=[],
        managers=None,
        *args, **kwargs
    ):
        """
        Initializes the PromptManager object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        """

        if len(goals) != len(targets):
            raise ValueError("Length of goals and targets must match")
        if len(goals) == 0:
            raise ValueError("Must provide at least one goal, target pair")

        self.tokenizer = tokenizer

        self._prompts = [
            managers['AP'](
                goal, 
                tokenizer, 
                conv_template, 
                targets[0], 
                control_init,
                test_prefixes
            )
            for goal in goals
        ]

        self._nonascii_toks = get_nonascii_toks(tokenizer, device='cpu')

    def generate(self, model, gen_config=None, find_target=False):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 64

        return [prompt.generate(model, gen_config, find_target) for prompt in self._prompts]
    
    def generate_str(self, model, gen_config=None, allow_watermark=False, find_target=False):
        if allow_watermark:
            return [
                self.tokenizer.decode(output_toks) 
                for output_toks in self.watermark_generate(model, gen_config)
            ]
        else:
            return [
                self.tokenizer.decode(output_toks) 
                for output_toks in self.generate(model, gen_config, find_target)
            ]
    
    def test(self, model, gen_config=None, find_target=False):
        return [prompt.test(model, gen_config, find_target) for prompt in self._prompts]

    def watermark_test(self, model, gen_config=None):
        return [prompt.watermark_test(model, gen_config) for prompt in self._prompts]

    def test_loss(self, model):
        return [prompt.test_loss(model) for prompt in self._prompts]
    
    def grad(self, model):
        return sum([prompt.grad(model) for prompt in self._prompts])
    
    def logits(self, model, test_controls=None, return_ids=False):
        vals = [prompt.logits(model, test_controls, return_ids) for prompt in self._prompts]
        if return_ids:
            return [val[0] for val in vals], [val[1] for val in vals]
        else:
            return vals
    
    def target_loss(self, logits, ids):
        return torch.cat(
            [
                prompt.target_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)
    
    def control_loss(self, logits, ids):
        return torch.cat(
            [
                prompt.control_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)
    
    def sample_control(self, *args, **kwargs):

        raise NotImplementedError("Sampling control tokens not yet implemented")

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, i):
        return self._prompts[i]

    def __iter__(self):
        return iter(self._prompts)
    
    @property
    def control_str(self):
        return self._prompts[0].control_str
    
    @property
    def control_toks(self):
        return self._prompts[0].control_toks

    @control_str.setter
    def control_str(self, control):
        for prompt in self._prompts:
            prompt.control_str = control
    
    @control_toks.setter
    def control_toks(self, control_toks):
        for prompt in self._prompts:
            prompt.control_toks = control_toks

    @property
    def target_str(self):
        return self._prompts[0].target_str
    
    @property
    def target_toks(self):
        return self._prompts[0].target_toks

    @target_str.setter
    def target_str(self, target):
        for prompt in self._prompts:
            prompt.target_str = target
    
    @target_toks.setter
    def target_toks(self, target_toks):
        for prompt in self._prompts:
            prompt.target_toks = target_toks

    @property
    def disallowed_toks(self):
        return self._nonascii_toks

class MultiPromptAttack(object):
    """A class used to manage multiple prompt-based attacks."""
    def __init__(self, 
        goals, 
        workers,
        targets=[],
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        *args, **kwargs
    ):
        """
        Initializes the MultiPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        workers : list of Worker objects
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list of str, optional
            The list of test goals of the attack
        test_targets : list of str, optional
            The list of test targets of the attack
        test_workers : list of Worker objects, optional
            The list of test workers used in the attack
        """

        self.goals = goals
        self.targets = targets  # not always default
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.test_prefixes = test_prefixes
        self.models = [worker.model for worker in workers]
        self.logfile = logfile
        self.prompts = [
            managers['PM'](
                goals,
                worker.tokenizer,
                worker.conv_template,
                targets,
                control_init,
                test_prefixes,
                managers
            )
            for worker in workers
        ]
        self.managers = managers
    
    @property
    def control_str(self):
        return self.prompts[0].control_str
    
    @control_str.setter
    def control_str(self, control):
        for prompts in self.prompts:
            prompts.control_str = control
    
    @property
    def control_toks(self):
        return [prompts.control_toks for prompts in self.prompts]
    
    @control_toks.setter
    def control_toks(self, control):
        if len(control) != len(self.prompts):
            raise ValueError("Must provide control tokens for each tokenizer")
        for i in range(len(control)):
            self.prompts[i].control_toks = control[i]

    @property
    def target_str(self):
        return self.prompts[0].target_str
    
    @target_str.setter
    def target_str(self, target):
        for prompts in self.prompts:
            prompts.targer_str = target
    
    @property
    def target_toks(self):
        return [prompts.target_toks for prompts in self.prompts]
    
    @target_toks.setter
    def target_toks(self, target):
        if len(target) != len(self.prompts):
            raise ValueError("Must provide control tokens for each tokenizer")
        for i in range(len(target)):
            self.prompts[i].target_toks = target[i]
    
    def get_filtered_cands(self, worker_index, control_cand, filter_cand=True, curr_control=None):
        cands, count = [], 0
        worker = self.workers[worker_index]
        # print('control cand shape:', control_cand.shape[0])
        for i in range(control_cand.shape[0]):
            decoded_str = worker.tokenizer.decode(control_cand[i], skip_special_tokens=True)
            # print('len of decoded str: ', len(worker.tokenizer(decoded_str, add_special_tokens=False).input_ids), 'len of control cand: ',  len(control_cand[i]))
            if filter_cand:
                # print('decoded str len: ', len(worker.tokenizer(decoded_str, add_special_tokens=False).input_ids), 'curr control len:', len(worker.tokenizer(curr_control, add_special_tokens=False).input_ids))
               
                if decoded_str != curr_control and len(worker.tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                    cands.append(decoded_str)                
                else:
                    count += 1
            else:
                cands.append(decoded_str)

        if cands == []:  ## check if cands is empty, if empty append again 
            for i in range(control_cand.shape[0]):
                decoded_str = worker.tokenizer.decode(control_cand[i], skip_special_tokens=True)
                cands.append(decoded_str)
 
        if filter_cand:
            # print([cands[-1]] * (len(control_cand) - len(cands)))
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
            # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
        return cands

    def step(self, *args, **kwargs):
        
        raise NotImplementedError("Attack step function not yet implemented")
    
    def run(self, 
        n_steps=100, 
        n_trial=1,
        batch_size=1024, 
        topk=256, 
        temp=1, 
        allow_non_ascii=True,
        target_weight=None, 
        control_weight=None,
        anneal=True,
        anneal_from=0,
        prev_loss=np.infty,
        stop_on_success=True,
        test_steps=50,
        log_first=False,
        filter_cand=True,
        verbose=True,
        fingerprint_success=False
    ):

        def P(e, e_prime, k):
            T = max(1 - float(k+1)/(n_steps+anneal_from), 1.e-7)
            return True if e_prime < e else math.exp(-(e_prime-e)/T) >= random.random()

        if target_weight is None:
            target_weight_fn = lambda _: 1
        elif isinstance(target_weight, (int, float)):
            target_weight_fn = lambda i: target_weight
        if control_weight is None:
            control_weight_fn = lambda _: 0.1
        elif isinstance(control_weight, (int, float)):
            control_weight_fn = lambda i: control_weight
        
        steps = 0
        loss = best_loss = 1e6
        best_control = self.control_str
        runtime = 0.
        success_cnt = 0.

        if self.logfile is not None and log_first:
            model_tests = self.test_all()
            self.log(anneal_from, 
                     n_steps+anneal_from, 
                     self.control_str, 
                     loss, 
                     runtime, 
                     model_tests, 
                     verbose=verbose)

        

        for i in range(n_steps): 

            if success_cnt < n_trial: 
                steps += 1
                start = time.time()
                torch.cuda.empty_cache()
                control, loss = self.step(
                    batch_size=batch_size, 
                    topk=topk, 
                    temp=temp, 
                    allow_non_ascii=allow_non_ascii, 
                    target_weight=target_weight_fn(i), 
                    control_weight=control_weight_fn(i),
                    filter_cand=filter_cand,
                    verbose=verbose
                )
                # print('after optiize new_control: ', len(control))
                runtime = time.time() - start
                keep_control = True if not anneal else P(prev_loss, loss, i+anneal_from)
                # print('keep control : ', keep_control)
                if keep_control:
                    self.control_str = control
                
                prev_loss = loss
                if loss < best_loss:
                    best_loss = loss
                    best_control = control

                print('Current Loss:', loss, 'Best Loss:', best_loss)

                if self.logfile is not None and (i+1+anneal_from) % test_steps == 0:
                    # print('self.control_str to last_constrol')
                    last_control = self.control_str
                    # print('best_control to self.control_str', len(self.control_str))
                    self.control_str = best_control
                    # print('after best_control to self.control_str: ', len(self.control_str))
                    model_tests = self.test_all()
                    control_str_id = self.workers[0].tokenizer(self.control_str).input_ids[1:]
                    fingerprint_success = self.log(i+1+anneal_from, n_steps+anneal_from, self.control_str, best_loss, runtime, model_tests, verbose=verbose)

                    self.control_str = last_control
            if stop_on_success:
                # model_tests = self.test(self.workers, self.prompts, include_loss=True)
                model_tests_jb, model_tests_mb, model_tests_loss, _ = model_tests
                # print('stop on success test: ', model_tests_jb ,model_tests_mb, model_tests_loss)
                if all(all(tests for tests in model_test) for model_test in model_tests_mb):
                   print(all(all(tests for tests in model_test) for model_test in model_tests_mb))
                   fingerprint_success = True
                    #    model_tests = self.test(self.workers, self.prompts)
                   _ = self.log(i+1+anneal_from, n_steps+anneal_from, self.control_str, best_loss, runtime, model_tests, verbose=verbose)
            if fingerprint_success and success_cnt <= n_trial:  
                success_cnt += 1
                fingerprint_success = False
            elif success_cnt > n_trial: break
            # print('after test best_control 3: ', len(self.control_str))
        return self.control_str, loss, steps, success_cnt

    def test(self, workers, prompts, include_loss=False, find_target=False):
        for j, worker in enumerate(workers):
            worker(prompts[j], "test", worker.model, None, find_target)
        model_tests = np.array([worker.results.get() for worker in workers])

        model_tests_jb = [[1] if jb == 'True' else 0 for jb in model_tests[..., 0]]
        model_tests_mb = model_tests[...,1].astype(int).tolist()
        model_tests_genstr = model_tests[...,2].astype(str).tolist()
        model_tests_loss = []
        if include_loss:
            for j, worker in enumerate(workers):
                worker(prompts[j], "test_loss", worker.model)
            model_tests_loss = [worker.results.get() for worker in workers]

        return model_tests_jb, model_tests_mb, model_tests_loss, model_tests_genstr

    def watermark_test(self, workers, prompts, include_loss=False):
        for j, worker in enumerate(workers):
            worker(prompts[j], "watermark_test", worker.model)
        model_tests = np.array([worker.results.get() for worker in workers])

        model_tests_jb = [[1] if jb == 'True' else 0 for jb in model_tests[..., 0]]
        model_tests_mb = model_tests[...,1].astype(int).tolist()
        model_tests_genstr = model_tests[...,2].astype(str).tolist()

        model_tests_loss = []
        if include_loss:
            for j, worker in enumerate(workers):
                worker(prompts[j], "test_loss", worker.model)
            model_tests_loss = [worker.results.get() for worker in workers]

        return model_tests_jb, model_tests_mb, model_tests_loss, model_tests_genstr

    def test_all(self):
        all_workers = self.workers + self.test_workers

        all_prompts = [
            self.managers['PM'](
                self.goals,
                worker.tokenizer,
                worker.conv_template,
                self.targets,
                self.control_str,
                self.test_prefixes,
                self.managers
            )
            for worker in all_workers
        ]
        return self.test(all_workers, all_prompts, include_loss=True)
    
    def parse_results(self, results):
        x = len(self.workers)
        i = len(self.goals)
        
        id_id = results[:x, :i].sum()
        id_od = results[:x, i:].sum()
        od_id = results[x:, :i].sum()
        od_od = results[x:, i:].sum()
        return id_id, id_od, od_id, od_od

    def log(self, step_num, n_steps, control, loss, runtime, model_tests, verbose=True):
        stop_on_success = True
        fingerprint_success = False
        jb, mb, loss, genstr = model_tests
        prompt_tests_jb, prompt_tests_mb, model_tests_loss = list(map(np.array, [jb, mb, loss]))
        # print('promtp_Test_mb:' , prompt_tests_mb)
        # print('model test loss: ', model_tests_loss)
        all_goal_strs = self.goals + self.test_goals
        all_workers = self.workers + self.test_workers
        tests = {
            'init: ' + all_goal_strs[i]:
            [
                (all_workers[j].model.name_or_path, prompt_tests_jb[j][i], prompt_tests_mb[j][i], model_tests_loss[j][i], genstr[j][i])
                for j in range(len(all_workers))
            ]
            for i in range(len(all_goal_strs))
        }
        n_passed = self.parse_results(prompt_tests_jb)
        n_em = self.parse_results(prompt_tests_mb)
        n_loss = self.parse_results(model_tests_loss)
        total_tests = self.parse_results(np.ones(prompt_tests_jb.shape, dtype=int))
        n_loss = [l / t if t > 0 else 0 for l, t in zip(n_loss, total_tests)]

        tests['control_str'] = control
        tests['n_passed'] = n_passed
        tests['n_em'] = n_em
        tests['n_loss'] = n_loss
        tests['total'] = total_tests

        with open(self.logfile, 'r') as f:
            log = json.load(f)
        
        log['controls'].append(control)
        log['losses'].append(n_loss[0])
        log['runtimes'].append(runtime)
        log['tests'].append(tests)

        with open(self.logfile, 'w', encoding='utf-8') as f: #change load json encoding format
            json.dump(log, f, indent=4, cls=NpEncoder, ensure_ascii=False)

        if verbose:
            output_str = ''
            for i, tag in enumerate(['id_id', 'id_od', 'od_id', 'od_od']):
                if total_tests[i] > 0:
                    output_str += f"({tag}) | Passed {n_passed[i]:>3}/{total_tests[i]:<3} | EM {n_em[i]:>3}/{total_tests[i]:<3} | Loss {n_loss[i]:.4f}\n"
            print((
                f"\n====================================================\n"
                f"Step {step_num:>4}/{n_steps:>4} ({runtime:.4} s)\n"
                f"{output_str}"
                f"Fingerprint='{control}'\n"
            ))
            if stop_on_success:
                for i, tag in enumerate(['id_id', 'id_od', 'od_id', 'od_od']):
                    if total_tests[i] > 0:
                        if n_em[i]//total_tests[i] == 1:
                            for j in range(len(all_workers)):
                                print('successfully fingerprint {}:', all_workers[j].model.name_or_path)
                            fingerprint_success = True
            print(f"====================================================\n")
        return fingerprint_success
       

class ProgressiveMultiPromptAttack(object):
    """A class used to manage multiple progressive prompt-based attacks."""
    def __init__(self, 
        goals, 
        workers,
        progressive_goals=True,
        progressive_models=True,
        targets=[],
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        test_prefixes=[],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        *args, **kwargs
    ):

        """
        Initializes the ProgressiveMultiPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        workers : list of Worker objects
            The list of workers used in the attack
        progressive_goals : bool, optional
            If true, goals progress over time (default is True)
        progressive_models : bool, optional
            If true, models progress over time (default is True)
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list of str, optional
            The list of test goals of the attack
        test_targets : list of str, optional
            The list of test targets of the attack
        test_workers : list of Worker objects, optional
            The list of test workers used in the attack
        """

        self.goals = goals
        self.targets = targets
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.progressive_goals = progressive_goals
        self.progressive_models = progressive_models
        self.control = control_init
        self.test_prefixes = test_prefixes
        self.logfile = logfile
        self.managers = managers
        self.mpa_kwargs = ProgressiveMultiPromptAttack.filter_mpa_kwargs(**kwargs)

        if logfile is not None:
            with open(logfile, 'w') as f:
                json.dump({
                        'params': {
                            'goals': goals,
                            'targets': targets,
                            'test_goals': test_goals,
                            'test_targets': test_targets,
                            'progressive_goals': progressive_goals,
                            'progressive_models': progressive_models,
                            'control_init': control_init,
                            'test_prefixes': test_prefixes,
                            'models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.workers
                            ],
                            'test_models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.test_workers
                            ]
                        },
                        'controls': [],
                        'losses': [],
                        'runtimes': [],
                        'tests': []
                    }, f, indent=4
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    def run(self, 
            n_steps: int = 1000, 
            n_trial: int = 1,
            batch_size: int = 1024, 
            topk: int = 256, 
            temp: float = 1.,
            allow_non_ascii: bool = False,
            target_weight = None, 
            control_weight = None,
            anneal: bool = True,
            test_steps: int = 50,
            incr_control: bool = True,
            stop_on_success: bool = True,
            verbose: bool = True,
            filter_cand: bool = True,
        ):
        """
        Executes the progressive multi prompt attack.

        Parameters
        ----------
        n_steps : int, optional
            The number of steps to run the attack (default is 1000)
        batch_size : int, optional
            The size of batches to process at a time (default is 1024)
        topk : int, optional
            The number of top candidates to consider (default is 256)
        temp : float, optional
            The temperature for sampling (default is 1)
        allow_non_ascii : bool, optional
            Whether to allow non-ASCII characters (default is False)
        target_weight
            The weight assigned to the target
        control_weight
            The weight assigned to the control
        anneal : bool, optional
            Whether to anneal the temperature (default is True)
        test_steps : int, optional
            The number of steps between tests (default is 50)
        incr_control : bool, optional
            Whether to increase the control over time (default is True)
        stop_on_success : bool, optional
            Whether to stop the attack upon success (default is True)
        verbose : bool, optional
            Whether to print verbose output (default is True)
        filter_cand : bool, optional
            Whether to filter candidates whose lengths changed after re-tokenization (default is True)
        """


        if self.logfile is not None:
            with open(self.logfile, 'r') as f:
                log = json.load(f)
                
            log['params']['n_steps'] = n_steps
            log['params']['n_trial'] = n_trial
            log['params']['test_steps'] = test_steps
            log['params']['batch_size'] = batch_size
            log['params']['topk'] = topk
            log['params']['temp'] = temp
            log['params']['allow_non_ascii'] = allow_non_ascii
            log['params']['target_weight'] = target_weight
            log['params']['control_weight'] = control_weight
            log['params']['anneal'] = anneal
            log['params']['incr_control'] = incr_control
            log['params']['stop_on_success'] = stop_on_success

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4)

        num_goals = 1 if self.progressive_goals else len(self.goals)
        num_workers = 1 if self.progressive_models else len(self.workers)
        step = 0
        stop_inner_on_success = self.progressive_goals
        loss = np.infty
        fingerprint_success=False

        while step < n_steps:
            attack = self.managers['MPA'](
                self.goals[:num_goals], 
                self.workers[:num_workers],
                self.targets[:num_goals],
                self.control,
                self.test_prefixes,
                self.logfile,
                self.managers,
                self.test_goals,
                self.test_targets,
                self.test_workers,
                **self.mpa_kwargs
            )
            #change stop on success setting
            if num_goals == len(self.goals) and num_workers == len(self.workers):
                stop_inner_on_success = True

            control, loss, inner_steps, fingerprint_success = attack.run(
                n_steps=n_steps-step,
                n_trial=n_trial,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                control_weight=control_weight,
                anneal=anneal,
                anneal_from=step,
                prev_loss=loss,
                stop_on_success=stop_inner_on_success,
                test_steps=test_steps,
                filter_cand=filter_cand,
                verbose=verbose, 
                fingerprint_success=fingerprint_success,
            )
            
            step += inner_steps
            self.control = control
            print('step: ', step, 'fingerprint sucess cnt: ', fingerprint_success)
            if fingerprint_success > n_trial:
                break

            # print('after optim control: ', len(self.control))

            if num_goals < len(self.goals):
                num_goals += 1
                loss = np.infty
            elif num_goals == len(self.goals):
                if num_workers < len(self.workers):
                    num_workers += 1
                    loss = np.infty
                elif num_workers == len(self.workers) and stop_on_success:
                    model_tests = attack.test_all()
                    # attack.log(step, n_steps, self.control, loss, 0., model_tests, verbose=verbose)
                    break
                else:
                    if isinstance(control_weight, (int, float)) and incr_control:
                        if control_weight <= 0.09:
                            control_weight += 0.01
                            loss = np.infty
                            if verbose:
                                print(f"Control weight increased to {control_weight:.5}")
                        else:
                            stop_inner_on_success = False
            # print('after testing control: ', len(self.control))
        return self.control, step

class IndividualPromptAttack(object):
    """ A class used to manage attacks for each target string / behavior."""
    def __init__(self, 
        goals, 
        targets,
        workers,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        *args,
        **kwargs,
    ):

        """
        Initializes the IndividualPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list
            The list of intended goals of the attack
        targets : list
            The list of targets of the attack
        workers : list
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list, optional
            The list of test goals of the attack
        test_targets : list, optional
            The list of test targets of the attack
        test_workers : list, optional
            The list of test workers used in the attack
        """

        self.goals = goals
        self.targets = targets
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.control = control_init
        self.control_init = control_init
        self.test_prefixes = test_prefixes
        self.logfile = logfile
        self.managers = managers
        self.mpa_kewargs = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)

        if logfile is not None:
            with open(logfile, 'w') as f:
                json.dump({
                        'params': {
                            'goals': goals,
                            'targets': targets,
                            'test_goals': test_goals,
                            'test_targets': test_targets,
                            'control_init': control_init,
                            'test_prefixes': test_prefixes,
                            'models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.workers
                            ],
                            'test_models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.test_workers
                            ]
                        },
                        'controls': [],
                        'losses': [],
                        'runtimes': [],
                        'tests': []
                    }, f, indent=4
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    def run(self, 
            n_steps: int = 1000, 
            batch_size: int = 1024, 
            topk: int = 256, 
            temp: float = 1., 
            allow_non_ascii: bool = True,
            target_weight: Optional[Any] = None, 
            control_weight: Optional[Any] = None,
            anneal: bool = True,
            test_steps: int = 50,
            incr_control: bool = True,
            stop_on_success: bool = True,
            verbose: bool = True,
            filter_cand: bool = True
        ):
        """
        Executes the individual prompt attack.

        Parameters
        ----------
        n_steps : int, optional
            The number of steps to run the attack (default is 1000)
        batch_size : int, optional
            The size of batches to process at a time (default is 1024)
        topk : int, optional
            The number of top candidates to consider (default is 256)
        temp : float, optional
            The temperature for sampling (default is 1)
        allow_non_ascii : bool, optional
            Whether to allow non-ASCII characters (default is True)
        target_weight : any, optional
            The weight assigned to the target
        control_weight : any, optional
            The weight assigned to the control
        anneal : bool, optional
            Whether to anneal the temperature (default is True)
        test_steps : int, optional
            The number of steps between tests (default is 50)
        incr_control : bool, optional
            Whether to increase the control over time (default is True)
        stop_on_success : bool, optional
            Whether to stop the attack upon success (default is True)
        verbose : bool, optional
            Whether to print verbose output (default is True)
        filter_cand : bool, optional
            Whether to filter candidates (default is True)
        """

        if self.logfile is not None:
            with open(self.logfile, 'r') as f:
                log = json.load(f)
                
            log['params']['n_steps'] = n_steps
            log['params']['test_steps'] = test_steps
            log['params']['batch_size'] = batch_size
            log['params']['topk'] = topk
            log['params']['temp'] = temp
            log['params']['allow_non_ascii'] = allow_non_ascii
            log['params']['target_weight'] = target_weight
            log['params']['control_weight'] = control_weight
            log['params']['anneal'] = anneal
            log['params']['incr_control'] = incr_control
            log['params']['stop_on_success'] = stop_on_success

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4)

        stop_inner_on_success = stop_on_success

        for i in range(len(self.goals)):
            print(f"Goal {i+1}/{len(self.goals)}")
            
            attack = self.managers['MPA'](
                self.goals[i:i+1], 
                self.targets[i:i+1],
                self.workers,
                self.control,
                self.test_prefixes,
                self.logfile,
                self.managers,
                self.test_goals,
                self.test_targets,
                self.test_workers,
                **self.mpa_kewargs
            )
            attack.run(
                n_steps=n_steps,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                control_weight=control_weight,
                anneal=anneal,
                anneal_from=0,
                prev_loss=np.infty,
                stop_on_success=stop_inner_on_success,
                test_steps=test_steps,
                log_first=True,
                filter_cand=filter_cand,
                verbose=verbose
            )

        return self.control, n_steps

class EvaluateAttack(object):
    """A class used to evaluate an attack using generated json file of results."""
    def __init__(self, 
        goals, 
        targets,
        workers,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        test_prefixes=[""],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        **kwargs,
    ):
        
        """
        Initializes the EvaluateAttack object with the provided parameters.

        Parameters
        ----------
        goals : list
            The list of intended goals of the attack
        targets : list
            The list of targets of the attack
        workers : list
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list, optional
            The list of test goals of the attack
        test_targets : list, optional
            The list of test targets of the attack
        test_workers : list, optional
            The list of test workers used in the attack
        """

        self.goals = goals
        self.targets = targets
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.control = control_init
        self.test_prefixes = test_prefixes
        self.logfile = logfile
        self.managers = managers
        self.mpa_kewargs = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)

        assert len(self.workers) == 1

        if logfile is not None:
            with open(logfile, 'w') as f:
                json.dump({
                        'params': {
                            'goals': goals,
                            'targets': targets,
                            'test_goals': test_goals,
                            'test_targets': test_targets,
                            'control_init': control_init,
                            'test_prefixes': test_prefixes,
                            'models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.workers
                            ],
                            'test_models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.test_workers
                            ]
                        },
                        'controls': [],
                        'losses': [],
                        'runtimes': [],
                        'tests': []
                    }, f, indent=4
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    @torch.no_grad()
    def run(self, steps, controls, batch_size, max_new_len=60, gen_config=None, verbose=True):

        model, tokenizer = self.workers[0].model, self.workers[0].tokenizer
        print('workers: ', self.workers[0].model)
        tokenizer.padding_side = 'left'

        if self.logfile is not None:
            with open(self.logfile, 'r') as f:
                log = json.load(f)

            log['params']['num_tests'] = len(controls)

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4)

        total_jb, total_em, total_outputs = [],[],[]
        test_total_jb, test_total_em, test_total_outputs = [],[],[]
        prev_control = 'haha'
        # for step, control in enumerate(controls):
        counter = 0
        for (mode, goals, targets) in zip(*[('Train', 'Test'), (self.goals, self.test_goals), (self.targets, self.test_targets)]):
            print(goals)
            print(targets)
            print('controls: ', controls)
            while counter < 1:
                if controls != prev_control and len(goals) > 0:
                    attack = self.managers['MPA'](
                        goals, 
                        self.workers,
                        targets,
                        controls,
                        self.test_prefixes,
                        self.logfile,
                        self.managers,
                        **self.mpa_kewargs
                    )
                    all_inputs = [p.eval_str for p in attack.prompts[0]._prompts]
                    max_new_tokens = [p.test_new_toks for p in attack.prompts[0]._prompts]
                    targets = [p.target for p in attack.prompts[0]._prompts]
                    all_outputs = []
                    # iterate each batch of inputs
                    for i in range(len(all_inputs) // batch_size + 1):
                        batch = all_inputs[i*batch_size:(i+1)*batch_size]
                        
                        batch_max_new = max_new_tokens[i*batch_size:(i+1)*batch_size]

                        batch_inputs = tokenizer(batch[0], return_tensors='pt')
                        # batch_inputs = tokenizer(batch, return_tensors='pt')
                        print(batch_inputs['input_ids'])
                        batch_str = tokenizer.decode(batch_inputs['input_ids'][0])
                        
                        batch_input_ids = batch_inputs['input_ids'].to(model.device)
                        # batch_attention_mask = batch_inputs['attention_mask'].to(model.device)
                        batch_attention_mask = torch.ones_like(batch_inputs['input_ids']).to(model.device)
                        
                        # position_ids = batch_attention_mask.long().cumsum(-1) - 1
                        # position_ids.masked_fill_(batch_attention_mask == 0, 1)
                        print('batch_str : ', batch_str)

                        if gen_config is None:
                            gen_config = model.generation_config
                            gen_config.max_new_tokens = max(max_new_len, max(batch_max_new))
                            # gen_config.temperature=0.9
                            # gen_config.do_sample=True
                            # gen_config.vocab_size=32000
                            # gen_config.max_new_tokens=9
                            gen_config.temperature=0.0
                            gen_config.top_p=0.95
                            gen_config.top_k=50
                            gen_config.typical_p=1
                            gen_config.repetition_penalty=1
                            gen_config.encoder_repetition_penalty=1
                            gen_config.no_repeat_ngram_size=0
                            gen_config.min_length=0
                            gen_config.tfs=1
                            gen_config.top_a=0
                            gen_config.penalty_alpha=0
                            gen_config.num_beans=1
                            gen_config.length_penalty=1
                            gen_config.mirostat_tau=5
                            gen_config.mirostat_eta=0.1
                            gen_config.do_sample=False
                            gen_config.suppress_tokens=[]
                            gen_config.vocab_size=32000
                            # gen_config.vocab_size=128000

                        print('model generation config: ', gen_config)

                        outputs = model.generate(batch_input_ids, attention_mask=batch_attention_mask, pad_token_id=tokenizer.pad_token_id, generation_config=gen_config)
                        # batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        batch_outputs = tokenizer.batch_decode(outputs)
                        
                        gen_start_idx = [len(tokenizer.decode(batch_input_ids[i])) for i in range(len(batch_input_ids))]
                        batch_outputs = [output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)]
                        print('output: ', batch_outputs)
                        all_outputs.extend(batch_outputs)

                        # clear cache
                        del batch_inputs, batch_input_ids, batch_attention_mask, outputs, batch_outputs
                        torch.cuda.empty_cache()
                    
                    curr_jb, curr_em = [], []
                    for (gen_str, target) in zip(all_outputs, targets):
                        jailbroken = not any([prefix in gen_str for prefix in self.test_prefixes])
                        em = target in gen_str
                        curr_jb.append(jailbroken)
                        curr_em.append(em)
                
                if mode == 'Train':
                    total_jb.append(curr_jb)
                    total_em.append(curr_em)
                    total_outputs.append(all_outputs)
                    # print(all_outputs)
                else:
                    test_total_jb.append(curr_jb)
                    test_total_em.append(curr_em)
                    test_total_outputs.append(all_outputs)

                # if verbose: print(f"{mode} Step {step+1}/{len(controls)} | Jailbroken {sum(curr_jb)}/{len(all_outputs)} | EM {sum(curr_em)}/{len(all_outputs)}")
                if verbose: print(f"{mode} | Jailbroken {sum(curr_jb)}/{len(all_outputs)} | EM {sum(curr_em)}/{len(all_outputs)}")

                # prev_control = control
                counter += 1


        return total_jb, total_em, test_total_jb, test_total_em, total_outputs, test_total_outputs


def set_env_variables(gpu_id):
    # Disable tensorflow logs, except in the case of an error.
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Set sharing strategy to file_system to avoid file descriptor leaks
    mp.set_sharing_strategy("file_system")

    # Only use one GPU, if we have one.
    # For Tensorflow
    # TODO: Using USE with `--parallel` raises similar issue as https://github.com/tensorflow/tensorflow/issues/38518#
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # For PyTorch
    torch.cuda.set_device(gpu_id)

    # Fix TensorFlow GPU memory growth
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                gpu = gpus[gpu_id]
                tf.config.experimental.set_visible_devices(gpu, "GPU")
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    except ModuleNotFoundError:
        pass


class ModelWorker(object):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **model_kwargs
        ).to(device).eval()

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue()
        self.process = None
        self.worker_pool = None


    @staticmethod
    def run(model, tasks, results):

        while True:
            task = tasks.get()

            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "watermark_test":
                        results.put(ob.watermark_test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

    @staticmethod
    def f(x):
        proc = mp.current_process()
        print('created process:', proc.name, proc._identity)
        return x * x

    @staticmethod  
    def run_from_queue(model, tasks, results, num_gpus, lock, first_to_start):

        proc = mp.current_process()
        print('created process:', proc.name, proc._identity)
        gpu_id = (mp.current_process()._identity[0] - 1) % num_gpus
        print('gpu id: ', gpu_id)
        set_env_variables(gpu_id)
        model.to(device)
        # Simple non-synchronized check to see if it's the first process to reach this point.
        # This let us avoid waiting for lock.
        if bool(first_to_start.value):
            # If it's first process to reach this step, we first try to acquire the lock to update the value.
            with lock:
                # Because another process could have changed `first_to_start=False` while we wait, we check again.
                if bool(first_to_start.value):
                    first_to_start.value = 0
        print('first to start: ', first_to_start.value)

        while True:
            task = tasks.get()

            if task is None:
                break
            ob, fn, args, kwargs = task
            if fn == "grad":
                with torch.enable_grad():
                    results.put(ob.grad(*args, **kwargs))
            else:
                with torch.no_grad():
                    if fn == "logits":
                        results.put(ob.logits(*args, **kwargs))
                    elif fn == "contrast_logits":
                        results.put(ob.contrast_logits(*args, **kwargs))
                    elif fn == "test":
                        results.put(ob.test(*args, **kwargs))
                    elif fn == "test_loss":
                        results.put(ob.test_loss(*args, **kwargs))
                    else:
                        results.put(fn(*args, **kwargs))
            tasks.task_done()

        

    def start_multigpu(self):

        num_gpus = torch.cuda.device_count()
        num_workers_per_device=1
        num_workers = num_workers_per_device * num_gpus
        print(f"Running {num_workers} worker(s) on {num_gpus} GPU(s).")

        # Lock for synchronization
        lock = mp.Lock()
        self.worker_pool = mp.Pool(num_workers, ModelWorker.run_from_queue, (self.model, self.tasks, self.results, num_gpus, lock, mp.Value("i", 1, lock=False)))
        # res = self.worker_pool.apply_async(ModelWorker.run_from_queue, args=(self.model, self.tasks, self.results, num_gpus, lock, mp.Value("i", 1, lock=False)))

        return self

    def start(self):
       
        self.process = mp.Process(
            target=ModelWorker.run,
            args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        print(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self
    
    def stop(self):
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self

    # def stop_multigpu(self):
    #     self.tasks.put(None)
    #     if self.worker_pool is not None:
    #         self.worker_pool.terminate()
    #         self.worker_pool.join()
    #     torch.cuda.empty_cache()
    #     return self

    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self

def get_workers(params, eval=False):
    tokenizers = []
    for i in range(len(params.tokenizer_paths)):
        tokenizer = AutoTokenizer.from_pretrained(
            params.tokenizer_paths[i],
            trust_remote_code=True,
            **params.tokenizer_kwargs[i]
        )
        if 'oasst-sft-6-llama-30b' in params.tokenizer_paths[i]:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        if 'guanaco' in params.tokenizer_paths[i]:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        if 'llama-2' in params.tokenizer_paths[i]:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'falcon' in params.tokenizer_paths[i]:
            tokenizer.padding_side = 'left'
        if 'mistral' in params.tokenizer_paths[i]:
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append(tokenizer)

    print(f"Loaded {len(tokenizers)} tokenizers")

    raw_conv_templates = [
        get_conversation_template(template)
        for template in params.conversation_templates
    ]
    print("param conversation template: ", params.conversation_templates)
    conv_templates = []
    for conv in raw_conv_templates:
        if conv.name == 'one_shot' and "zero_shot" in params.conversation_templates:
            conv.roles = tuple(['### ' + r for r in conv.roles])
            conv.sep = " "
        elif conv.name == 'llama-2':
            conv.sep2 = conv.sep2.strip()
        elif conv.name == 'alpaca':
            conv.set_system_message("Below is an instruction that describes a task. Write a response that appropriately completes the request.")
            conv.sep = " "
        elif conv.name == 'one_shot' and "chatgpt" in params.conversation_templates:
            conv.set_system_message("You are a helpful assistant.")
            conv.roles = tuple(["user", "assistant"])
            conv.sep = " "
        # elif 'vicuna' in conv.name :
        #     print('set system message!!!')
        #     conv.set_system_message("A chat between a curious human and an artificial intelligence assistant.\n"
        #     "The assistant gives helpful, detailed, and polite answers to the human’s questions.\n")
        conv_templates.append(conv)
        
    print(f"Loaded {conv.name}, {len(conv_templates)} conversation templates")
    workers = [
        ModelWorker(
            params.model_paths[i],
            params.model_kwargs[i],
            tokenizers[i],
            conv_templates[i],
            params.devices[i]
        )
        for i in range(len(params.model_paths))
    ]
    if not eval:
        for worker in workers:
            worker.start()

    num_train_models = getattr(params, 'num_train_models', len(workers))
    print('Loaded {} train models'.format(num_train_models))
    print('Loaded {} test models'.format(len(workers) - num_train_models))

    return workers[:num_train_models], workers[num_train_models:]

def get_goals_and_targets(params):

    train_goals = getattr(params, 'goals', [])
    train_targets = getattr(params, 'targets', [])
    test_goals = getattr(params, 'test_goals', [])
    test_targets = getattr(params, 'test_targets', [])
    offset = getattr(params, 'data_offset', 0)

    if params.train_data:
        train_data = pd.read_csv(params.train_data)
        train_targets = train_data['target'].tolist()[offset:offset+params.n_train_data]
        if 'goal' in train_data.columns:
            train_goals = train_data['goal'].tolist()[offset:offset+params.n_train_data]
        else:
            train_goals = [""] * len(train_targets)
        if params.test_data and params.n_test_data > 0:
            test_data = pd.read_csv(params.test_data)
            test_targets = test_data['target'].tolist()[offset:offset+params.n_test_data]
            if 'goal' in test_data.columns:
                test_goals = test_data['goal'].tolist()[offset:offset+params.n_test_data]
            else:
                test_goals = [""] * len(test_targets)
        elif params.n_test_data > 0:
            test_targets = train_data['target'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
            if 'goal' in train_data.columns:
                test_goals = train_data['goal'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
            else:
                test_goals = [""] * len(test_targets)

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    print('Loaded {} train goals'.format(len(train_goals)))
    print('Loaded {} test goals'.format(len(test_goals)))

    return train_goals, train_targets, test_goals, test_targets
