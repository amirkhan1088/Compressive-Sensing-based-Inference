function m = sigma_delta_UD_Counter_col_new(x,ref)
        % x is the grey image    
        % ref is the reference signal for comparator
        % m is number of samples corresponding to each column
        
        % this function will be in the loop for all columns of image

        x = double(x);
        nc = size(x,2); % number of columns in the image
        nr = size(x,1); % number of rows in the image
        col = zeros(nc,1); % matrix storing the samples columnwise
        f0=255; % maximum signal to be feedback to input depending on the 
        % comparator output
        aref = ref; %127,255,0;  %comparator reference voltage
        for j=1:nc
            x1 = x(:,j); 
            a0 = 0;
            f=0; %feedback signal
            Dig_out = zeros(nr,1); % digital equivalent of each column 
            k=0; 
            for i=1:nr
                a = x1(i) - f; % summing difference of input and feedback signal
                a1 = a+a0; % ----> accumulator's present output
                a0 = a1;
                c = a1-aref; % output of the comparator
                % 1-bit quantizer which gives +1 (255 for images) and -1 (-255) as the output
                if c>=0
                    f = f0; %=255 if images feedback signal
                    Dig_out(i) = k +1;
                    k = Dig_out(i);
                else
                    f=-f0; %=-255 if images
                    Dig_out(i) = k-1;
                    k = Dig_out(i); % k-1
                end
            end
            col(j) = max(Dig_out);
        end
        %m = sum(col);
        m = col;
end