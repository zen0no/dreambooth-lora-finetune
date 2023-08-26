import os
import torch
import pandas as pd
from code import ClipCriteria

from PIL import Image

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='EvaluateModel',
                                    description='Evaluate diffusion model')

    parser.add_argument('-D', '--data_path')
    parser.add_argument('-g', '--generated_path')
    parser.add_argument('-o', '--output_path')
    parser.add_argument('-d', '--device')

    args = parser.parse_args()

    criteria = ClipCriteria(device=args.device)
    data_path = args.data_path
    generated_path = args.genereated_path

    data = pd.read_csv(os.path.join(data_path, 'classes.csv'))

    results = {
        'id_': [],
        'prompt_fidelty': [],
        'image_fidelty': [] ,
    }

    for i, row in data.iterrows():
        id_,  token, class_name = row.id, row.token, row.class_

        prompts_path = os.path.join(data_path, 'prompts', f'{id_}.txt')
        prompts = [x.format(token, class_name).rstrip('\n') for x in open(prompts_path, 'r').readlines()]

        reference_image_path =  os.path.join(data_path, 'images', id_, '0.jpg')
        reference_image = Image.open(reference_image_path)

        prompt_fidelty = []
        image_fidelty = []

        for i, prompt in enumerate(prompts):
            image_path = os.path.join(generated_path, id_, f'{i}.jpg')
            generated_image = Image.open(image_path)

            clipI = criteria.clipI(generated_image, prompt)
            clipT = criteria.clipT(generated_image, prompt)

            prompt_fidelty.append(clipT)
            image_fidelty.append(clipI)

        results['id_'].append(id_)
        results['prompt_fidelty'].append(sum(prompt_fidelty) /
                                         len(prompt_fidelty))
        results['image_fidelty'].append(sum(image_fidelty) /
                                        len(prompt_fidelty))

        results = pd.DataFrame(results)
