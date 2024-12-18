 %filename = strcat('results/ObjBefore25.txt');
 %filename = strcat('AAA_ONLY-BEFORE-TwoNormResults.txt');
 %filename = strcat('results/ObjBefore25.txt');
 %filename = strcat('AAA_ONLY-AFTER-TwoNormResults.txt');
 filename = strcat('facebookResults/FacebookObjBefore.txt');
 fid = fopen(filename, 'r' ); % changed 
 formatSpec = '%f';
 sizeA = [1 Inf];
 A = fscanf(fid,formatSpec,sizeA);
 fclose(fid);

 %filename = strcat('results/ObjAfter25.txt');
 %filename = strcat('results/ObjBefore25.txt');
 %filename = strcat('AAA_ONLY-AFTER-TwoNormResults.txt');
 %filename = strcat('Initial_CF_SS.txt');
 %filename = strcat('results/ObjAfter25.txt');
 %filename = strcat('Final_CF_SS.txt');
 %filename = strcat('Steady_state_initial.txt');
 %filename = strcat('Steady_state_final.txt');
 filename = strcat('facebookResults/facebookObjAfter.txt');

 fid = fopen(filename, 'r' ); % changed 
 formatSpec = '%f';
 sizeA = [1 Inf];
 B = fscanf(fid,formatSpec,sizeA);
 fclose(fid);
    
 range = 100;

 B = B(1:range);
 A = A(1:range);
 x = (A-B);
 y = zeros(range,1);
 test = zeros(range,1);
 fileB = 'PercDiff.txt';

 fidResultB = fopen(fileB, 'a' );
 for i = 1:range
    y(i) = abs((x(i) / A(i))) * 100; 
    fprintf( fidResultB, '%s\n', num2str(y(i)) );

 end


 %fclose(fidResultB);

 boxplot(y)
 %histogram(y)
 %ylabel('% Difference')
 %ytickangle(90) 
 mean1 = sum(y) / range;
 x = median(y);
 %{
 y1 = uint32([1:length(A)]);
 plot(y1,A,'LineWidth',1)
 hold on
 plot(y1,B)
 legend('C++ Simulation','MatLab CF')
 mean1 = sum(y) / 100;
 %}