function PlotSPS(v,nn)
    n = length(v);
    v = v([1:50*nn (n-150*nn+1):n]);
    figure('Renderer', 'painters', 'Position', [800 500/nn 400 200])
    axes('Position', [0.07 0.09 0.89 0.88]);
    
    pt= plot(1:length(v),zeros(length(v),1), '-', 'LineWidth', 1); 
    
    pt(1).Color='#f26419'; hold on; %0e6fbb
    stem(find(v), v(v ~= 0), 'o-',  'filled','Color','#1c8ddb', 'MarkerSize',6, 'LineWidth', 1);
    
    grid on;
    
    ymin = min(v(v ~= 0));
    ymax = max(v(v ~= 0));
    y    = max(abs(ymin),abs(ymax));
    axis([1 length(v) -1.05*y 1.05*y]);
    
    if nn == 1
        xticks([1, 50, 150, 200]);
        xticklabels({'1','50', '450','500'});
    else
        xticks([1, 250, 750, 1000]);
        xticklabels({'1','250', '2250','2500'});
    end
    
end

