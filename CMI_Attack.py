import numpy as np 
import torch
import torch.nn as nn
import copy
from torchvision import transforms
from nltk.corpus import wordnet2021
from gensim.models import KeyedVectors
glovemodel = KeyedVectors.load('glove2word2vev300d.model')

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class CMIAttacker():
    def __init__(self, model, multi_attacker):
        self.model = model
        self.multi_attacker = multi_attacker

    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        momentum = torch.zeros_like(imgs).detach().to(device)
        # N
        N_steps = 1
        adv_txts, momentum = self.multi_attacker.txt_attack(self.model, txts, imgs, txt2img, N_steps, momentum, scales=scales)
        # M
        M_steps = 10
        adv_imgs = imgs
        adv_imgs = self.multi_attacker.img_attack(self.model, adv_txts, adv_imgs, imgs, txt2img, M_steps, momentum, device, scales=scales)
        return adv_imgs, adv_txts
    
class Attack():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30,number_perturbation=1, topk=10, threshold_pred_score=0.3, imgs_eps=2/255, step_size=0.5/255, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.cls = cls
        self.imgs_eps = imgs_eps 
        self.step_size = step_size
        self.batch_size = batch_size


    def txt_attack(self, net, texts,imgs,txt2img,steps,momentum, scales=None):
        device = self.ref_net.device
        b, _, _, _ = imgs.shape
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1
        images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        imgs_0 = imgs.detach() + torch.from_numpy(np.random.uniform(-self.imgs_eps, self.imgs_eps, imgs.shape)).float().to(device)
        imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)
        scaled_imgs = self.get_scaled_imgs(imgs_0, scales, device)     

        for _ in range(steps):
            final_adverse = []
            origin_output = net.inference_text(text_inputs)
            if self.cls:
                origin_embeds = origin_output['text_feat'][:, 0, :].detach()
            else:
                origin_embeds = origin_output['text_feat'].flatten(1).detach()

            imgs_0.requires_grad_()
            imgs_output = net.inference_image(images_normalize(imgs_0))
            imgs_embeds = imgs_output['image_feat'][txt2img]
            net.zero_grad()

            txtloss = torch.tensor(0.0, dtype=torch.float32).to(device)
            for i, text in enumerate(texts):
                important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)
                list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
                words, sub_words, keys = self._tokenize(text)
                final_words = copy.deepcopy(words)
                change = 0
                
                for top_index in list_of_index:
                    if change >= self.num_perturbation:
                        break
                    tgt_word = words[top_index[0]]
                    if tgt_word in filter_words:
                        continue
                    if keys[top_index[0]][0] > self.max_length - 2:
                        continue
                    substitutes = []

                    # Embedding Guidance
                    try:
                        similar_words = glovemodel.most_similar(tgt_word)
                        substitutes = [word for word, _ in similar_words]
                    except KeyError:

                        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
                        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k
                        substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k

                        word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                        substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                                    self.threshold_pred_score)
                        pass

                    replace_texts = [' '.join(final_words)]
                    available_substitutes = [tgt_word]
                    for substitute_ in substitutes:
                        substitute = substitute_

                        if substitute == tgt_word:
                            continue  # filter out original word
                        if '##' in substitute:
                            continue  # filter out sub-word
                        if '.' in substitute:
                            continue  # filter out .
                        if substitute in filter_words:
                            continue

                        temp_replace = copy.deepcopy(final_words)
                        temp_replace[top_index[0]] = substitute
                        available_substitutes.append(substitute)
                        replace_texts.append(' '.join(temp_replace))
                    replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                    replace_output = net.inference_text(replace_text_input)
                    if self.cls:
                        replace_embeds = replace_output['text_feat'][:, 0, :]
                    else:
                        replace_embeds = replace_output['text_feat'].flatten(1)
                    txtloss = self.loss_func(replace_embeds, imgs_embeds, i)
                    candidate_idx = txtloss.argmax()

                    final_words[top_index[0]] = available_substitutes[candidate_idx]
                    if available_substitutes[candidate_idx] != tgt_word:
                        change += 1

                final_adverse.append(' '.join(final_words))


            txts_input = self.tokenizer(final_adverse, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
            txts_output = net.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
            imgs_output = net.inference_image(images_normalize(scaled_imgs))
            imgs_embeds = imgs_output['image_feat']
            with torch.enable_grad():
                img_loss_list = []
                img_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    img_loss_item = self.loss_func1(imgs_embeds[i*b:i*b+b], txt_supervisions, txt2img)
                    img_loss_list.append(img_loss_item.item())
                    img_loss += img_loss_item

            loss = img_loss
            loss.backward()

            grad = imgs_0.grad
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
            grad = grad + momentum * 0.9
            momentum = grad
            perturbation = self.step_size * grad.sign()
            imgs_0 = imgs_0.detach() + perturbation
            imgs_0 = torch.min(torch.max(imgs_0, imgs - self.imgs_eps), imgs + self.imgs_eps)
            imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)

        return final_adverse, momentum
    
    def img_attack(self, model, texts, imgs,origin_imgs,txt2img,steps,momentum,device, scales=None):
        model.eval()
        b, _, _, _ = imgs.shape
        
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1

        images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        imgs_0 = imgs.detach() + torch.from_numpy(np.random.uniform(-self.imgs_eps, self.imgs_eps, origin_imgs.shape)).float().to(device)
        imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)

        txts_input = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
        txts_output = model.inference_text(txts_input)
        txt_supervisions = txts_output['text_feat'].detach()

        for j in range(steps):
            imgs_0.requires_grad_()
            model.zero_grad()
            scaled_imgs = self.get_scaled_imgs(imgs_0, scales, device)     
            imgs_output = model.inference_image(images_normalize(scaled_imgs))
            imgs_embeds = imgs_output['image_feat']
            with torch.enable_grad():
                img_loss_list = []
                img_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    img_loss_item = self.loss_func1(imgs_embeds[i*b:i*b+b], txt_supervisions, txt2img)
                    img_loss_list.append(img_loss_item.item())
                    img_loss += img_loss_item
            loss =  img_loss 
            loss.backward()
            grad = imgs_0.grad
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)           
            grad = grad + momentum * 0.9
            momentum = grad
            perturbation = self.step_size * grad.sign()
            imgs_0 = imgs_0.detach() + perturbation
            imgs_0 = torch.min(torch.max(imgs_0, origin_imgs - self.imgs_eps), origin_imgs + self.imgs_eps)
            imgs_0 = torch.clamp(imgs_0, 0.0, 1.0)

        return imgs_0

    def synonym_augment(self, word):
        synonyms = set()
        for syn in wordnet2021.synsets(word):
            for lem in syn.lemmas(): 
                synonym = lem.name()
                if synonym != word:
                    synonyms.add(synonym)

        return list(synonyms)
    
    def loss_func1(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1

        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1) 
        loss = loss_TaIcpos
        return loss

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)



def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words