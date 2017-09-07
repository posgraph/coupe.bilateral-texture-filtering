function [M, gpuM] = comp_MRTV_gpu(L, fr)

eps_gm = 1e-09;

gx = imfilter(L, [0 -1 1], 'symmetric');
gy = imfilter(L, [0 -1 1].', 'symmetric');
gm = hypot(gx, gy);
% gm = sum(hypot(gx, gy), 3);

% gm = comp_Delta(L, 1);

p_gm = single(padarray(gm, [fr fr], 'symmetric'));

[h, w, d] = size(L);
pu = fr+1;
pb = pu+h-1;
pl = fr+1;
pr = pl+w-1;

gpuM = gpuArray(zeros(h, w, d, 'single'));
gpu_eps = gpuArray(ones(h, w, 'single')*eps_gm);

for i = 1:d
%   gi = gpuArray(p_gm(:, :, i));
  gi = p_gm(:, :, i);
  max_gm = gpuArray(zeros(h, w, 'single'));
  sum_gm = gpuArray(zeros(h, w, 'single'));
  
  for y = -fr:fr
    for x = -fr:fr
      max_gm = max(max_gm, gi(pu+y:pb+y, pl+x:pr+x));
      sum_gm = sum_gm + gi(pu+y:pb+y, pl+x:pr+x);
    end
  end
  
  sum_gm = max(sum_gm, gpu_eps);
  gpuM(:, :, i) = max_gm./sum_gm * (2*fr+1);
%   gpuM(:, :, i) = sum_gm;
end

M = mean(gpuM, 3);
% M = log1p(M);

% max_gm = gpuArray(zeros(h, w, 'single'));
% sum_gm = gpuArray(zeros(h, w, 'single'));
% 
% for y = -fr:fr
%   for x = -fr:fr
%     max_gm = max(max_gm, p_gm(pu+y:pb+y, pl+x:pr+x));
%     sum_gm = sum_gm + p_gm(pu+y:pb+y, pl+x:pr+x);
%   end
% end
% 
% gpuM = max_gm./(sum_gm+eps_gm);
% 
% M = gpuM * (2*fr+1);

end