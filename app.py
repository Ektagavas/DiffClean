import gradio as gr
import numpy as np
import subprocess

import cv2
import os



def process_image(image, modeltype, estimate_age):
    
    img = cv2.imread(image)
    cv2.imwrite('./temp.png', img)
    
    if modeltype == 'ssr':
        modelpath = 'checkpoint/diffclean_ssr_age.pth'
    elif modeltype == 'clip':
        modelpath = 'checkpoint/diffclean_clip_age.pth'

    print(modelpath)

    print('Processing uploaded image...')
    
    cmd = f"python main.py --edit_one_image_MR --config UTK.yml --exp ./runs_2204/gradio_sample --n_iter 1 --t_0 80 --n_inv_step 40 --n_train_step 6 --n_test_step 6 --img_path ./temp.png --model_path {modelpath} --schedule cosine"

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    

    stdout, stderr = process.communicate()
    if process.returncode != 0: print(stderr.decode())
    print("Processing complete...")

    prefix = modelpath.split('/')[1].split('.pth')[0]

    op_path = f"./output/gradio_sample_E1_MR_t80_UTK_gtemp_t80_ninv40_{prefix}/image_samples/3_gen_t80_it0_gtemp_ninv40_ngen6_mrat1_{prefix}.png"

    print(op_path)
    
    if estimate_age:
        print('Estimating Age...')
        process = subprocess.Popen(f"python3 ../MiVOLO/demo.py --input {op_path} --output 'output' --detector-weights ../MiVOLO/models/yolov8x_person_face.pt --checkpoint ../MiVOLO/models/model_imdb_cross_person_4.22_99.46.pth.tar --device 'cuda:0' --draw", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        print(stdout)
        if process.returncode != 0: print(stderr.decode())
        print("Processing complete...")

        op_path = f"./output/out_3_gen_t80_it0_gtemp_ninv20_ngen6_mrat1_{prefix}.jpg"
        
    op = cv2.cvtColor(cv2.imread(op_path), cv2.COLOR_BGR2RGB)
    return op



demo = gr.Interface(process_image, [gr.Image(type='filepath'),gr.Dropdown(
            ["ssr","clip"], label="Model", info="Choose DiffClean model"
        ), "checkbox"], "image")
    
if __name__ == "__main__":
    demo.launch(share=True)  # Share your demo with just 1 extra parameter 