global FILE_COUNT;
global TOTALCT;
global CREATED_FLAG;

%string='train';
string='valid';
if strcmp(string, 'train') == 1

    %hyper_dir = '../dataset/veins_t34bands/train_data/mat/';
    hyper_dir = 'D://PhD_Projects//HSI_Project//vein-visualization-master//Visualization_and_Testing//train_data//mat//';
    label=dir(fullfile(hyper_dir,'*.mat'));

    rgb_dir = 'D://PhD_Projects//HSI_Project//vein-visualization-master//Visualization_and_Testing//train_data//rgb//';
    order= randperm(size(label,1));   
   
else

    hyper_dir = 'D://PhD_Projects//HSI_Project//vein-visualization-master//Visualization_and_Testing//valid_data//mat//';
    label=dir(fullfile(hyper_dir,'*.mat'));
    
    rgb_dir = 'D://PhD_Projects//HSI_Project//vein-visualization-master//Visualization_and_Testing//valid_data//rgb//';
    order= randperm(size(label,1)); % It is randomizing the label values
    
end  

%% Initialization the patch and stride
% size_input=50;
% size_label=50;
size_input=64;
size_label=64;
label_dimension=3;
data_dimension=3;
stride=64;


%% Initialization the hdf5 parameters
prefix=[string '_t32bands'];
chunksz=64;
TOTALCT=0;
FILE_COUNT=0;
amount_hd5_image=50000;
CREATED_FLAG=false;

disp("size(label,1)")
disp(size(label,1));

%% For loop  RGB-HS-HD5  
for i=1:size(label,1)
     if mod(i,amount_hd5_image)==1     
         filename = getFilename(label(order(i)).name, prefix, hyper_dir);
     end
    name_label=strcat(hyper_dir,label(order(i)).name);
    
    a_temp=struct2cell(load(name_label,'rad')); % struct2cell converts the structure array to cell array. 
    % structure array contains homogeneous data type only, whereas the cell
    % array can different data types.
    hs_label=cell2mat(a_temp);
%     hs_label=hs_label/(2^12-1);
    rgb_name=[ rgb_dir label(order(i)).name(1:end-4) '.jpg'];
    
    rgb_data_uint=imread(rgb_name);
    %rr = double(rgb_data_uint)
    rgb_data=im2double(rgb_data_uint); % convert all pixel values to 0 to 1 range
    %rgb_data = zscore(rr);
    
    %for j=1:label_dimension
    
%     ConvertHStoNbands(rgb_data,hs_label(:,:,3),size_input,size_label,1,data_dimension,stride,chunksz,amount_hd5_image,filename)
    %end
    %for j=1:3:16
    %    ConvertHStoNbands(rgb_data,hs_label(:,:,j:j+2),size_input,size_label,label_dimension,data_dimension,stride,chunksz,amount_hd5_image,filename)
    %end
    ConvertHStoNbands(rgb_data,hs_label,size_input,size_label,label_dimension,data_dimension,stride,chunksz,amount_hd5_image,filename)
end       
 

function filename_change=getFilename(filename,prefix, folder_label)
       filename_change=filename;
       filename_change=[prefix filename_change];
       filename_change=filename_change(1:end-4);
       filename_change=strcat(filename_change,'.h5');
end

