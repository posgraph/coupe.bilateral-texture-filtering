function r_img = blf_2d_gpu(img, G, fr, sigma_avg)

global o_img

[h, w, ~] = size(img);
p_G = padarray(G, [fr fr], 'symmetric');
p_img = padarray(img, [fr fr], 'symmetric');

r_img = gpuArray(zeros(size(img), 'single'));
sum_d_W = gpuArray(zeros(h, w, 'single'));
eps_d_W = gpuArray(ones(h, w, 'single')*eps);

pu = fr+1;
pb = pu+h-1;
pl = fr+1;
pr = pl+w-1;

[x,y] = meshgrid(-fr:fr,-fr:fr);
SW = exp(-(x.^2+y.^2)/(2*fr^2));

sigma_l = 1/3*sqrt(size(img, 3));

for x = -fr:fr
    for y = -fr:fr
        d_W = p_G(pu+y:pb+y, pl+x:pr+x, :) - G;
        d_W = sum(d_W.^2, 3);
        d_W = exp(-0.5 * (d_W/sigma_avg/sigma_avg));
        d_W = d_W * SW(fr+y+1, fr+x+1);  % Gaussian weight
        d_L = p_img(pu+y:pb+y, pl+x:pr+x, :) - o_img;
        d_L = sum(d_L.^2, 3);
        d_L = exp(-0.5 * (d_L/sigma_l/sigma_l));
        d_W = d_W.*d_L;
        
        sum_d_W = sum_d_W + d_W;
        r_img = r_img + bsxfun(@times, d_W, p_img(pu+y:pb+y, pl+x:pr+x, :));
    end
end

r_img = bsxfun(@rdivide, r_img, max(sum_d_W, eps_d_W));

end
