load('att_splits.mat')
trainval_loc=union(train_loc, val_loc)
trainval_loc=trainval_loc(randperm(length(trainval_loc)))
save att_splits.mat att original_att test_seen_loc test_unseen_loc trainval_loc train_loc val_loc
