T=readtable('UCM_img.csv', 'Delimiter', ',');
image_files=T{:,1};
Y=readtable('UCM_label.csv', 'Delimiter', ',', 'Format', '%f');
labels=Y{:,1};
save('res101.mat','image_files', 'labels')
