function plotCompleteGraph(A,max_weight)
    n = length(A);

    % Plot a circle with some radius
    radius = 1;
    angle = 0:0.01:2*pi;
    plot(radius*cos(angle),radius*sin(angle),'blue');
    hold on

    % Plot n nodes around the circle (distributed evenly)
    step = 2*pi/n;
    for i=1:n
        scatter(radius*cos(i*step),radius*sin(i*step),100,'filled','o','MarkerEdgeColor','black','MarkerFaceColor','yellow')
    end
    
    % Plot the edges with color or transparency depending on the weight
    for i=1:n
        for j=i+1:n
            edge_x = [radius*cos(i*step),radius*cos(j*step)];
            edge_y = [radius*sin(i*step),radius*sin(j*step)];
            alpha_value = 1 - A(i,j)/max_weight;
            plot(edge_x,edge_y,'-','Color',[0,0,0,alpha_value],'LineWidth',1.5);
        end
    end
end

