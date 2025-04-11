from random import random
import torch
from fusion.mm_utils import tokenizer_image_token
def call_fusion_engine_df(args, sample, model, tokenizer=None, processor=None):
    from fusion.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from fusion.conversation import conv_templates, SeparatorStyle

    def deal_with_prompt(input_text, mm_use_im_start_end):
        mm_use_im_start_end = True
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    instructs = tokenizer(prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()).input_ids
    instructs = torch.tensor(instructs, dtype=torch.long).unsqueeze(0).cuda()
    if image is not None:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image.unsqueeze(0).half().cuda(),
                do_sample=False,
                # temperature=0,
                # top_p=None,
                # num_beams=5,
                max_new_tokens=128,
                use_cache=True,
                instructs=instructs)

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def fusion_image_processor(raw_image, vis_processors=None):
    image_tensor = vis_processors.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    return image_tensor