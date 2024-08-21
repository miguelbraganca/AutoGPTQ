import numpy as np
import torch.nn as nn
from datasets import load_dataset
import torch, time
from tqdm import tqdm
import random
import gc
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_wikitext2_v2(model : torch.nn.Module, tokenizer, max_length=1024, stride=512, verbose=True):
    set_seed(42)
    model = model.to('cuda')
    model.eval()
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    encodings = tokenizer('\n\n'.join(dataset['text']), return_tensors='pt')

    lls, t = [], []
    for i in tqdm(range(0, encodings['input_ids'].size(1), stride), disable=not verbose):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings['input_ids'].size(1))
        trg_len = end_loc - i
        input_ids = encodings['input_ids'][:,begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100 # ignore context

        t1 = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs.loss * trg_len
                print(f'log_likelihood: {log_likelihood.item()}')  # Print loss values

        torch.cuda.synchronize()
        t2 = time.time()
        t.append((t2-t1))
        lls.append(log_likelihood)

        del input_ids, target_ids

    total_loss = torch.stack(lls).sum()
    ppl = np.round(float(torch.exp(total_loss / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    if verbose:
        print('perplexity:', ppl)
        print('time:', str(pred_time) + ' sec')

    del encodings
    torch.cuda.empty_cache()  # Ensure memory is cleaned up

    return ppl, pred_time


# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


@torch.no_grad()
def opt_eval(model, testenc, dev, seqlen=2048):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit-1g"

def main():
    traindataset, _ = get_wikitext2(128, 0, 2048, pretrained_model_dir)

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act = False,  # desc_act and group size only works on triton
        sym = False
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset, use_triton=False)

    # save/load quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)

    eval_wikitext2_v2(model, tokenizer, verbose=True)
    #opt_eval(model.model, testenc, "cuda:0")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
