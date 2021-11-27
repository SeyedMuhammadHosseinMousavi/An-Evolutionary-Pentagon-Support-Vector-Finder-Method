function PlotSolution(X, sol)

    % Cluster Centers
    m = sol.Out.m;
    k = size(m,1);
    
    % Cluster Indices
    ind = sol.Out.ind;
    
    Colors = hsv(k);
    
    for j=1:k
        Xj = X(ind==j,:);
        plot(Xj(:,1),Xj(:,2),'x','LineWidth',2,'Color',Colors(j,:));
        hold on;
    end
    
    plot(m(:,1),m(:,2),'ok','LineWidth',2,'MarkerSize',12);
    
    hold off;
    
end