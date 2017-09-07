I = imread('input.png');
radius = 2;
iterations = 5;
radius_bf = radius * 2;
S = btf_2d_color_gpu(I, radius, iterations, radius_bf);
figure;
imshow(S);
