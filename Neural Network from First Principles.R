library(colorspace)
library(glmnet)
library(kableExtra)
color.gradient = function(x, colors=c('magenta','white','lightblue'), colsteps=50)
{
  colpal = colorRampPalette(colors)
  return( colpal(colsteps)[ findInterval(x, seq(min(x),max(x), length=colsteps)) ] )
}
dat = read.table('SpamBotData_2024.txt', h = TRUE, stringsAsFactors = TRUE)
attach(dat)

# Part A

# 1 for robot, 0 for human
dat$Spam_Bot <- as.character(dat$Spam_Bot)
dat$Spam_Bot[dat$Spam_Bot == "Robot"] <- 1
dat$Spam_Bot[dat$Spam_Bot == "Human"] <- 0
dat$Spam_Bot <- as.numeric(dat$Spam_Bot)

dat$Lure_Flag <- as.numeric(dat$Lure_Flag)
dat$Sentiment <- as.numeric(dat$Sentiment)

X = as.matrix(dat[,2:5])
Y = as.matrix(dat[,1])
U = as.matrix(X[,1:2])
V = as.matrix(X[,3:4])
kable(head(dat,4), caption = "First four rows of the data.")


# Part B

# Obj fn. 

g = function(AL,Y)
{
  obj <- 0
  AL <- pmax(pmin(AL, 1 - 1e-7), 1e-7)
  for(i in 1:length(Y)) {
    
    obj_i <- (Y[i]*log(AL[i])) + ((1-Y[i])*log(1-AL[i]))
    obj <- obj + obj_i
  }
  obj <- -obj / length(Y)
  return(obj)
}

# Part C


# Part D
## create NN architect

#tanh activation
sig1 = function(z)
{
  tanh(z)
}

#sigmoid activation
sig2 = function(z) {
  1 / (1 + exp(-z))
}

N <- nrow(U)  
p <- ncol(U)  
q <- ncol(V)  

neural_net = function(U,V,Y,theta,m,nu)
{
  N <- nrow(U)
  p <- ncol(U)  
  q <- ncol(V)  
  
  index <- 1
  
  X <- cbind(U, V)
  # number of variables
  n <- p + q
  
  npars=(m*p*2+1)+(m*2+1)+(2*1+1)
  
  theta = runif(npars,-1,1)
  # put weights in
  W1u <- matrix(theta[index:(index + m * p - 1)], nrow=p, ncol=m)
  index <- index + m * p
  
  W1v <- matrix(theta[index:(index + m * q - 1)], nrow=q, ncol=m)
  index <- index + m * q
  
  # Construct the weight matrix W as a nx2m matrix
  W <- matrix(0, nrow=n, ncol=2*m)
  #partition W
  W[1:p, 1:m] <- W1u
  W[(p+1):(p+q), (m+1):(m*2)] <- W1v
  
  # Bias terms
  b_1 <- rep(theta[index], each = N * m)
  b_1_matrix <- matrix(b_1, nrow = N , ncol = 2*m, byrow = F)
  index <- index + 1
  
  Z1 <- X %*% W + b_1_matrix
  A1 <- tanh(Z1)
  
  W2u <- matrix(theta[index:(index + m-1 )], nrow=m, ncol=p-1)
  index <- index + m
  
  W2v <- matrix(theta[index:(index + m-1 )], nrow=m, ncol=q-1)
  index <- index + m
  
  W2 <- matrix(0, nrow=2*m, ncol=p)
  W2[1:m, p-1] <- W2u
  W2[(m+1):(2*m), p] <- W2v
  
  b_2 <- rep(theta[index], each = N)
  b_2_matrix <- matrix(b_2, nrow = N , ncol = q, byrow = F)
  index <- index + 1
  
  Z2 <- A1 %*% W2 + b_2_matrix
  A2 <- tanh(Z2)
  
  w_0 <- matrix(theta[index:(index + 2 - 1 )], nrow=2, ncol=1)
  index <- index + 2
  
  b_0 <- rep(theta[index], each = N)
  index <- index + 1
  b_0_matrix <- matrix(b_0, nrow = N , ncol = 1, byrow = F)
  
  Z3 <- A2 %*% w_0 + b_0_matrix
  out <- sig2(Z3)
  
  E1 <- g(out, Y)
  E2 <- nu *  sum(abs(theta))
  
  objective_function <- E1 + E2
  return(list(out = out, E1 = E1, E2 = E2))
}


m     = 5
npars=(m*p*2+1)+(m*2+1)+(2*1+1)
theta_rand = runif(npars,-1,1)

obj <- function(theta){
  res <- neural_net(U,V,Y,theta,m,0)
  #loss function
  E1 <- res$E1
  #regularization penalty
  E2 <- res$E2
  return(E1 + E2)
  
}

# Part E
# peform CV to find regularization parameter

set.seed(2024)

row_indices <- 1:nrow(dat)
train_indices <- sample(row_indices, size = floor(nrow(dat) * 0.8))
train_data <- dat[train_indices, ]
validation_data <- dat[-train_indices, ]

# training data
X_train = as.matrix(train_data[,2:5])
Y_train = as.matrix(train_data[ ,1])
U_train <- as.matrix(X_train[,1:2])
V_train <- as.matrix(X_train[,3:4])

# validation data
X_validation = as.matrix(validation_data[,2:5])
Y_validation = as.matrix(validation_data[ ,1])
U_validation <- as.matrix(X_validation[,1:2])
V_validation <- as.matrix(X_validation[,3:4])

# range of nu values to test
nu_values <- exp(seq(-10,0,length = 50))

obj_train <- function(theta) {
  res <- neural_net(U_train, V_train, Y_train, theta, m, 0)
  return(res$E1 + res$E2)  
}

validation_errors <- numeric(length(nu_values))
for (i in seq_along(nu_values)) {
  theta_rand = runif(npars,-1,1)
  # Optimize the model on the training set
  opt_res <- optim(par = theta_rand, fn = obj_train, method = "CG")
  
  # Evaluate the error on the validation set
  train_res <- neural_net(U_train, V_train, Y_train, opt_res$par, m, 0)
  validation_result <- neural_net(U_validation, V_validation, Y_validation,
                                  opt_res$par, m, 0)
  # Only the error, not the regularization
  validation_errors[i] <- validation_result$E1 
}

# Find the nu with the lowest validation error
best_nu <- round(nu_values[which.min(validation_errors)],3)
best_error <- round(min(validation_errors),3)

# Plot the validation error vs. nu
plot(nu_values, validation_errors, ylim = c(0, max(validation_errors)),
     type = "l", main = " ",
     xlab = "Regularization Parameter (nu)", ylab = "Validation Error")


# optimize parameters
obj_final <- function(theta){
  res <- neural_net(U,V,Y,theta,m,best_nu)
  E1 <- res$E1
  E2 <- res$E2
  return(E1 + E2)
  
}
opt_res <- optim(par = theta_rand, fn = obj, method = "CG")
opt_res_mag <- optim(par = theta_rand, fn = obj_final, method = "CG")
# use abs value as we are plotting magnitude
plot(abs(opt_res_mag$par), type = 'h', lwd = 2, col = 'blue',
     main = " ",
     xlab = "Parameter Index",
     ylab = "Magnitude")

# Part G
## create response curves

set.seed(123)
N = 400

sent_constant <- 1
y_col <- matrix(c(rep(0,N)),ncol =1)

# lure 1
data_lure_1 <- dat[dat$Lure_Flag==1,]
#create possible values for continuous variables
grammar_score_range_1 <- seq(min(data_lure_1[,2]), max(data_lure_1[,2]), length = N)
emoji_score_range_1 <- seq(min(data_lure_1[,3]), max(data_lure_1[,3]), length = N)

x1_gram_range_1 <- rep(grammar_score_range_1, N)
x2_emoji_range_1 <- rep(emoji_score_range_1, each = N)
v1_lure_cons_1 <- rep(1, length = length(x1_gram_range_1))
v2_sent_cons_1 <- rep(sent_constant, length = length(x1_gram_range_1))


Lat <- as.matrix(data.frame(Grammar_Score= x1_gram_range_1, Emoji_Score = x2_emoji_range_1))
Lat_V <- as.matrix(data.frame(v1_lure_cons_1, v2_sent_cons_1))

# feed data into the NN
res_pred <- neural_net(Lat, Lat_V, y_col,opt_res$par,m,best_nu)
res_out <- res_pred$out

plot(x2_emoji_range_1~x1_gram_range_1,col = color.gradient(res_out),pch = 16,
     cex = 0.5, xlab = "Grammer", ylab = "Emoji")
text(data_lure_1$Grammar_Score,data_lure_1$Emoji_Score, labels = as.numeric(data_lure_1$Spam_Bot))


# lure 2
data_lure_2 <- dat[dat$Lure_Flag==2,]
#create possible values for continuous variables
grammar_score_range_2 <- seq(min(data_lure_2[,2]), max(data_lure_2[,2]), length = N)
emoji_score_range_2 <- seq(min(data_lure_2[,3]), max(data_lure_2[,3]), length = N)

x1_gram_range_2 <- rep(grammar_score_range_2, N)
x2_emoji_range_2 <- rep(emoji_score_range_2, each = N)
v1_lure_cons_2 <- rep(2, length = length(x1_gram_range_2))
v2_sent_cons_2 <- rep(sent_constant, length = length(x1_gram_range_2))

Lat_2 <- as.matrix(data.frame(Grammar_Score= x1_gram_range_2, Emoji_Score = x2_emoji_range_2))
Lat_V_2 <- as.matrix(data.frame(v1_lure_cons_2, v2_sent_cons_2))

# feed data into the NN
res_pred_2 <- neural_net(Lat_2, Lat_V_2, y_col,opt_res$par,m,best_nu)
res_out_2 <- res_pred_2$out

plot(x2_emoji_range_2~x1_gram_range_2,col = color.gradient(res_out_2),pch = 16,
     cex = 0.5, xlab = "Grammer", ylab = "Emoji")
text(data_lure_2$Grammar_Score,data_lure_2$Emoji_Score, labels = as.numeric(data_lure_2$Spam_Bot))

# lure 3
data_lure_3 <- dat[dat$Lure_Flag==3,]
#create possible values for continuous variables
grammar_score_range_3 <- seq(min(data_lure_3[,2]), max(data_lure_3[,2]), length = N)
emoji_score_range_3 <- seq(min(data_lure_3[,3]), max(data_lure_3[,3]), length = N)

x1_gram_range_3 <- rep(grammar_score_range_3, N)
x2_emoji_range_3 <- rep(emoji_score_range_3, each = N)
v1_lure_cons_3 <- rep(3, length = length(x1_gram_range_3))
v2_sent_cons_3 <- rep(sent_constant, length = length(x1_gram_range_3))


Lat_3 <- as.matrix(data.frame(Grammar_Score= x1_gram_range_3, Emoji_Score = x2_emoji_range_3))
Lat_V_3 <- as.matrix(data.frame(v1_lure_cons_3, v2_sent_cons_3))

# feed data into the NN
res_pred_3 <- neural_net(Lat_3, Lat_V_3, y_col,opt_res$par,m,best_nu)
res_out_3 <- res_pred_3$out

plot(x2_emoji_range_3~x1_gram_range_3,col = color.gradient(res_out_3),pch = 16,
     cex = 0.5, xlab = "Grammer", ylab = "Emoji")
text(data_lure_3$Grammar_Score,data_lure_3$Emoji_Score, labels = as.numeric(data_lure_3$Spam_Bot))