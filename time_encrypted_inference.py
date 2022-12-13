import numpy as np
import tenseal as ts

from utils import *
from models import *
from activations import *
from encrypted_models import *


if __name__ == "__main__":

    polynomial_orders = np.arange(1,5)
    model_names = ["MLPNet", "LeNet"]
    activations = ["LeakyRelu", "Sigmoid", "SoftplusX"]

    for model_name in model_names:
        for activation in activations:
            for order in polynomial_orders:
                if model_name == "MLPNet":
                    net = MLPNet()
                else:
                    net = LeNet()

                checkpoint = torch.load(f'checkpoint/{model_name}/{order}/ckpt_{activation}.pth')
                net.load_state_dict(checkpoint['net'])

                # Initialize encrypted model
                if model_name == "MLPNet":
                    enc_model = EncMLPNet(net, activation, order)
                else:
                    enc_model = EncLeNet(net, activation, order)

                if activation == "LeakyRelu":
                    enc_model.change_all_activations(leakyReluX(order=order))
                elif activation == "Sigmoid":
                    enc_model.change_all_activations(sigmoidX(order=order))
                else:
                    enc_model.change_all_activations(softplusX(order=order))

                # Get test data and create testloader
                testloader = get_test_loader(data_dir='./data',
                                             batch_size=25,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=False)


                images, labels = next(iter(testloader))

                bits_scale = 26

                # Create TenSEAL context
                context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=8192*2,
                    coeff_mod_bit_sizes=[31]+[bits_scale]*14+[31]
                    )

                # set the scale
                context.global_scale = pow(2, bits_scale)

                # galois keys are required to do ciphertext rotations
                context.generate_galois_keys()

                # Determine if we want to benchmark
                with open(f"checkpoint/{model_name}/{order}/stats_{activation}.txt", "r") as f:
                    lines = f.readlines()
    
                benchmark = True
                for line in lines:
                    if "nan" in line:
                        print("Not benchmarking due to NaN")
                        benchmark = False

                if benchmark:
                    inference_times = []
                    for i in range(len(images)):
                        if model_name == "MLPNet":
                            enc_img = ts.ckks_vector(context, images[i].squeeze().flatten().tolist())
                            start = time.time()
                            enc_out = enc_model(enc_img)
                            end = time.time()
                        else:
                            kernel_shape = net.convL0_.kernel_size
                            stride = net.convL0_.stride[0]
                            enc_img, windows_nb = ts.im2col_encoding(context, images[i].view(28,28).tolist(), kernel_shape[0], kernel_shape[1], stride)
                            start = time.time()
                            enc_out = enc_model(enc_img, windows_nb)
                            end = time.time()

                        inf_time = (end-start)*1000
                        print(round(inf_time,2))
                        inference_times.append(inf_time)

                    average_ms_inference = np.mean(np.array(inference_times))
                    with open(f"checkpoint/{model_name}/{order}/stats_{activation}.txt", "w") as f:
                        for line in lines:
                            if not line.startswith("Time"):
                                f.write(line)
                        print(f"Writing mean inference time to file: {round(average_ms_inference,2)} ms.")
                        f.write(f"Time (ms): {average_ms_inference}") 
