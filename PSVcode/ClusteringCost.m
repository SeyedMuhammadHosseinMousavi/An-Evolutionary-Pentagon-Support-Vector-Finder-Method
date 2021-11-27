function [z, out] = ClusteringCost(s, X, Method)

    if ~exist('Method','var')
        Method = 'DB';
    end

    m = s(:,1:end-1);
    
    a = s(:,end);
    if sum(a>=0.5)<2
        [~, SortOrder] = sort(a,'descend');
        a(SortOrder(1:2)) = 1;
    end
    
    m = m(a>=0.5,:);
    
    switch Method
        case 'DB'
            [z, out] = DBIndex(m, X);
            out.m = m;
            
        case 'CS'
            [z, out] = CSIndex(m, X);
            
    end
    
    out.a = double(a>=0.5);
    
end