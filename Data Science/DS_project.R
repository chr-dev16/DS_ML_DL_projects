df = read.csv("online_shoppers_intention.csv")
set.seed(71)

df$TrafficType = as.factor(df$TrafficType)
df$VisitorType = as.factor(df$VisitorType)
df$Browser = as.factor(df$Browser)
df$Region = as.factor(df$Region)
df$OperatingSystems = as.factor(df$OperatingSystems)
df$Month = as.factor(df$Month)
df$Weekend = as.factor(df$Weekend)
df$Revenue = as.factor(df$Revenue)
y = df$Revenue

dmy = dummyVars(~ . , data = df)   #one-hot encoding for categorical variables
df = data.frame(predict(dmy, newdata = df))
df$Revenue.FALSE = NULL
df$Revenue.TRUE = NULL
df$Revenue = y

num_features = data.frame(df$Administrative, df$Administrative_Duration, df$Informational, df$Informational_Duration,
                          df$ProductRelated, df$ProductRelated_Duration, df$BounceRates, df$ExitRates, df$PageValues, df$SpecialDay)
num_features_scaled = data.frame(scale(num_features))
drops = c('Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay')
df_cat = df[,!(names(df) %in% drops)]
df_cat$ID = seq.int(nrow(df))
num_features_scaled$ID = seq.int(nrow(num_features_scaled))
df = join(df_cat, num_features_scaled, by = "ID")

df = df[sample(nrow(df)),]
size_train = 0.8*nrow(df)      # 80/20 train-test-split
start_test = size_train + 1
df_train = df[1:size_train,]
df_test = df[start_test:12330,]

df_train_reduced = data.frame(x = data.frame(df_train$df.PageValues, df_train$df.ExitRates, df_train$df.ProductRelated_Duration), y = df_train$Revenue)
df_test_reduced = data.frame(x = data.frame(df_test$df.PageValues, df_test$df.ExitRates, df_test$df.ProductRelated_Duration), y = df_test$Revenue)
names(df_train_reduced) = c("x.PageValues", "x.ExitRates", "x.ProductRelated_Duration", "y")
names(df_test_reduced) = c("x.PageValues", "x.ExitRates", "x.ProductRelated_Duration", "y")
y_true = df_test_reduced[,ncol(df_test_reduced)]

size_train = 0.8*nrow(df_train_reduced)     #extract validation set
start_val = size_train + 1
df_val = df_train_reduced[start_val:9864,]
df_train_reduced = df_train_reduced[1:size_train,]


#Random Forest

f1_rf_val = hash()
ntree_grid = c(20,30,50,80,120,160,200,500,600)
for (number_of_trees in ntree_grid){
  rf = randomForest(y ~ . , data = df_train_reduced, ntree = number_of_trees, importance = TRUE)
  rf_pred = predict(rf, df_val[,1:3])
  f1_rf_val[number_of_trees] = F1_Score(df_val[,4], rf_pred, positive = "TRUE")
}
f1_rf_val = as.list(f1_rf_val)
best_ntree = as.numeric(names(which.max(f1_rf_val)))
rf = randomForest(y ~ . , data = df_train_reduced, ntree = best_ntree, importance = TRUE)
rf_pred = predict(rf, df_test_reduced[,1:3])
f1_rf = F1_Score(y_true, rf_pred, positive = "TRUE")

#SVM

f1_svm_val = hash()
gamma_grid = c(0.1,0.33,0.5,0.8)
for (gamma in gamma_grid){
  SVC = svm(y ~ . , data = df_train_reduced, kernel = "radial", gamma = gamma)
  svm_pred = predict(SVC, df_val[,1:3])
  f1_svm_val[gamma] = F1_Score(df_val[,4], svm_pred, positive = "TRUE")
}
f1_svm_val = as.list(f1_svm_val)
best_gamma = as.numeric(names(which.max(f1_svm_val)))
SVC = svm(y ~ . , data = df_train_reduced, kernel = "radial", gamma = best_gamma)
svm_pred = predict(SVC, df_test_reduced[,1:3])
f1_svm = F1_Score(y_true, svm_pred, positive = "TRUE")

#Log_Reg

f1_logreg_val = hash()
logreg = glm(y ~ . , data = df_train_reduced, family = binomial(link = "logit"))
y_prob = predict(logreg, df_test_reduced[,1:3], type = 'response')
y_true_logreg = as.integer(y_true)
cutoff_grid = c(0.3, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.45)
for (cutoff in cutoff_grid){
  y_pred = ifelse(y_prob > cutoff, 2, 1)
  f1_logreg_val[cutoff] = F1_Score(y_true_logreg, y_pred, positive = "TRUE")
}
f1_logreg_val = as.list(f1_logreg_val)
best_cutoff = as.numeric(names(which.max(f1_logreg_val)))
y_pred = ifelse(y_prob > best_cutoff, 2, 1)
f1_logreg = F1_Score(y_true_logreg, y_pred, positive = "TRUE")

#Log_Reg with LASSO regularization

lambdas = 10^seq(0, -3, by = -.1)
logreg_lasso_cv = cv.glmnet(as.matrix(df_train_reduced[,1:3]), df_train_reduced[,4], family = "binomial", lambda = lambdas)
opt_lambda = logreg_lasso_cv$lambda.min
logreg_lasso = glmnet(as.matrix(df_train_reduced[,1:3]), df_train_reduced[,4], family = "binomial", lambda = opt_lambda)

f1_logreg_lasso_val = hash()
y_prob = predict(logreg_lasso, as.matrix(df_test_reduced[,1:3]), type = 'response')
y_true_logreg = as.integer(y_true)
cutoff_grid = c(0.3, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.45)
for (cutoff in cutoff_grid){
  y_pred = ifelse(y_prob > cutoff, 2, 1)
  f1_logreg_lasso_val[cutoff] = F1_Score(y_true_logreg, y_pred, positive = "TRUE")
}
f1_logreg_lasso_val = as.list(f1_logreg_lasso_val)
best_cutoff = as.numeric(names(which.max(f1_logreg_lasso_val)))
y_pred_lasso = ifelse(y_prob > best_cutoff, 2, 1)
f1_logreg_lasso = F1_Score(y_true_logreg, y_pred_lasso, positive = "TRUE")

#KNN

x_train = df_train_reduced[,1:3]
y_train = df_train_reduced[,4]
x_test = df_test_reduced[,1:3]
f1_knn_val = hash()
K_grid = c(1,3,4,8,10,30,35,40,50,75,150)
for (K in K_grid){
  knn_pred = knn(x_train, x_test, y_train, k = K)
  f1_knn_val[K] = F1_Score(y_true, knn_pred, positive = "TRUE")
}
f1_knn_val = as.list(f1_knn_val)
best_K = as.numeric(names(which.max(f1_knn_val)))
knn_pred = knn(x_train, x_test, y_train, k = best_K)
f1_knn = F1_Score(y_true, knn_pred, positive = "TRUE")

#LDA

lda.fit = lda(y ~ . , data = df_train_reduced)
lda.fit
lda.pred = predict(lda.fit, newdata = df_test_reduced)
lda.class = lda.pred$class
f1_lda = F1_Score(y_true, lda.class, positive = "TRUE")
f1_lda

qda.fit = qda(y ~ . , data = df_train_reduced)
qda.fit
qda.pred = predict(qda.fit, newdata = df_test_reduced)
qda.class = qda.pred$class
f1_qda = F1_Score(y_true, qda.class, positive = "TRUE")
f1_qda

f1_scores = hash()
f1_scores["RF"] = f1_rf
f1_scores["SVM"] = f1_svm
f1_scores["Log Reg"] = f1_logreg
f1_scores["Log Reg LASSO"] = f1_logreg_lasso
f1_scores["KNN"] = f1_knn
f1_scores["LDA"] = f1_lda
accuracies = hash()
accuracies["RF"] = Accuracy(rf_pred, y_true)
accuracies["SVM"] = Accuracy(svm_pred, y_true)
accuracies["Log Reg"] = Accuracy(y_pred, y_true_logreg)
accuracies["Log Reg LASSO"] = Accuracy(y_pred_lasso, y_true_logreg)
accuracies["KNN"] = Accuracy(knn_pred, y_true)
accuracies["LDA"] = Accuracy(lda.class, y_true)
f1_scores
accuracies