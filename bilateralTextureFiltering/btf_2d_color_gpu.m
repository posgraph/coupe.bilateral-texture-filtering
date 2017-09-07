function r_img = btf_2d_color_gpu(I, fr, n_iter, fr_blf)
%   btf_2d_color_gpu - Bilateral Texture Filtering
%
%   S = btf_2d_color_gpu(I, fr, n_iter, fr_blf) extracts structure S from
%    input I, with scale parameter fr, joint filtering scale fr_blf and
%    iteration number n_iter. 
%   
%   Paras: 
%   @I         : Input image, both grayscale and color images are acceptable.
%   @fr        : Parameter specifying the maximum size of texture elements.                    
%   @n_iter    : Number of itearations, 5 by default.
%   @fr_blf    : Parameter specifying kernel size of joint bilateral filtering.  
%            
%   Example
%   ==========
%   I = imread('input.png');
%   radius = 3;
%   iterations = 3;
%   radius_bf = radius * 2;
%   S = btf_2d_color_gpu(I, radius, iterations, radius_bf);
%
%   ==========
%   The Code is created based on the following paper 
%   [1] "Bilateral Texture Filtering", Hojin Cho, Hyunjoon Lee, Seungyong Lee, ACM Transactions on Graphics, 
%   (SIGGRAPH 2014), 2014. 
%   The code and the algorithm are for non-comercial use only.
%

global o_img

if ~exist('fr_blf', 'var') || isempty(fr_blf),
  fr_blf = 2*fr;
end

if ~exist('n_iter', 'var') || isempty(n_iter),
  n_iter = 5;
end

sigma_avg = 0.05*sqrt(size(I, 3));
sigma_alpha = 5;

tic;

I = gpuArray(im2single(I));
o_img = I;

for iter = 1:n_iter
  fprintf('iter = %d\n', iter);
  L = I;
  Gc = cell(fr, 1);
  %Lcpu = gather(L);
  for i = fr:fr
    
    B = imfilter(L, fspecial('average', 2*i+1), 'symmetric');
  
    % MRTV
    Delta = comp_Delta_gpu(L, i);
    M = comp_MRTV_gpu(L, i);
    M = mean(M.*Delta, 3); 

    % comp_S
    [S, M_min, ~] = comp_S(B, M, i);

    % alpha blending
    M_diff = M - M_min;

    alpha = sigmoid(M_diff, sigma_alpha);
    alpha = (alpha - 0.5) * 2;    
    alpha = repmat(alpha, [1 1 size(S, 3)]);

    G = (alpha).*S + (1-alpha).*B;

    Gc{i} = G;
  end
  
  G = Gc{end};
  r_img = blf_2d_gpu(I, G, fr_blf, sigma_avg);
  I = r_img;
end

r_img = gather(I);

et = toc;
disp(['elapsed time = ' num2str(et)]);

end


function b = sigmoid(a, p)

b = 1 ./ (1 + exp(-p.*a));

end

% comp_S
function [S, M_min, min_idx] = comp_S(B, M, fr)

[h, w, d] = size(B);

p_M = padarray(M, [fr fr], 'replicate');
p_B = padarray(B, [fr fr], 'symmetric');
pu = fr+1;
pb = pu+h-1;
pl = fr+1;
pr = pl+w-1;

% minimum value
S = B; %gpuArray(zeros(size(B), 'single'));
M_min = M; %gpuArray(ones(h, w, 'single')*1000); % arbitrary large value
oX = gpuArray(zeros(h, w, 'single'));
oY = gpuArray(zeros(h, w, 'single'));

min_idx = gpuArray(reshape(1:h*w, [h w]));
p_min_idx = padarray(min_idx, [fr fr], 'symmetric');

for x = -fr:fr
  for y = -fr:fr
    n_M = p_M(pu+y:pb+y, pl+x:pr+x);
    n_B = p_B(pu+y:pb+y, pl+x:pr+x, :);
    n_min_idx = p_min_idx(pu+y:pb+y, pl+x:pr+x);
    
    idx = n_M < M_min;
    M_min = min(M_min, n_M);
    
    oX = oX.*(1-idx) + x.*idx;
    oY = oY.*(1-idx) + y.*idx;
    
    min_idx = n_min_idx.*idx + min_idx.*(1-idx);
    idx = repmat(idx, [1 1 d]);
    S = n_B.*idx + S.*(1-idx);
  end
end

end


