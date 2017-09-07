function [Delta, gpuDelta] = comp_Delta_gpu(L, fr)

p_L = single(padarray(L, [fr fr], 'symmetric'));
[h, w, d] = size(L);
pu = fr+1;
pb = pu+h-1;
pl = fr+1;
pr = pl+w-1;
gpuDelta = gpuArray(zeros(h, w, d, 'single'));
for i = 1:d;
%   Li = gpuArray(p_L(:, :, i));
  Li = p_L(:, :, i);
  max_L = gpuArray(zeros(h, w, 'single'));
  min_L = gpuArray(ones(h, w, 'single'));
  
  for y = -fr:fr
    for x = -fr:fr
      max_L = max(max_L, Li(pu+y:pb+y, pl+x:pr+x));
      min_L = min(min_L, Li(pu+y:pb+y, pl+x:pr+x));
    end
  end
  
  gpuDelta(:, :, i) = max_L - min_L;
end

Delta = mean(gpuDelta, 3);

end