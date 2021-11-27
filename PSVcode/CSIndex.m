function [CS, out] = CSIndex(m, X)

    k = size(m,1);
    n = size(X,1);
    
    % Calculate Distance Matrix
    d = pdist2(X, m);
    
    % Assign Clusters and Find Closest Distances
    [dmin, ind] = min(d, [], 2);
    
    dmax = zeros(n,1);
    dxx = pdist2(X,X);
    for p=1:n
        dmax(p) = max(dxx(p,ind==ind(p)));
    end
    
    dbar = zeros(k,1);
    for i=1:k
        if sum(ind==i)>0
            dbar(i) = mean(dmax(ind==i));
            m(i,:) = mean(X(ind==i,:));
        else
            dbar(i) = 10*norm(max(X)-min(X));
        end
    end
    
    D=pdist2(m,m);
    for i=1:k
        D(i,i)=inf;
    end
    Dmin=min(D);
    
    CS = mean(dbar)/mean(Dmin);
    
    out.d=d;
    out.dmin=dmin;
    out.ind=ind;
    out.CS=CS;
    out.dmax=dmax;
    out.dbar=dbar;
    out.D=D;
    out.Dmin=Dmin;
    out.m=m;
    
end