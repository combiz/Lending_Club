# ==============================================================================
# ==============================================================================
# Data Science Modelling Exercise for Lending Club Data
# January 2018
# Objective: Build a logistic regression model on previously unseen data using R.
# by Combiz Khozoie, Ph.D.
# ==============================================================================

#   ____________________________________________________________________________
#   Load packages and data                                                  ####

##  ............................................................................
##  Load packages                                                           ####

require(readr)
require(data.table)
require(ggplot2)
require(pROC)
require(gmodels)
require(InformationValue)
require(caret)
require(rpart)
require(glmulti)
require(ROSE)
require(MLmetrics)
require(caret)
require(ROSE)
require(broom)
require(InformationValue)
library(plotly)
require(hmeasure)
library(ggplot2)
library(extrafont)
font_import()
loadfonts(device = "win")

##  ............................................................................
##  Load data                                                               ####

# A dataset derived from the Lending Club (https://www.lendingclub.com/).
# The data contains rows of customers, with each column showing the features
# for loan applications that have been approved, together with outcomes of the
# loans (in the final column).  The outcomes show that each customer has either
# defaulted or completed their loan.
lc <- data.table(read_csv("LendingClub.csv"))
tidy.names <- make.names(names(lc), unique=TRUE) 
names(lc) <- tidy.names  # remove spaces from column names etc

# Data attribute definitions were obtained from 
# the Data Dictionary (https://resources.lendingclub.com/LCDataDictionary.xlsx)

##  ............................................................................
##  Preliminary data preparation                                            ####

#count NAs
colSums(is.na(lc))

# the number of NAs is miniscule relative to sample size, ok to drop
lc <- lc[complete.cases(lc), ]

# create unique id to facilitate some wrangling operations
lc[, id := .I]
setkey(lc, key = id)

# create new bClass as binary 0 / 1 of Class (default = 0)
lc$bClass <- 0
lc[Class == 'Creditworthy', bClass := 1]
lc$Class <- NULL # drop the original Class
str(lc)

#   ____________________________________________________________________________
#   Data Cleanup and Preparation                                            ####

##  ............................................................................
##  Categorical numbers to numeric                                          ####

## Some text based numbers are suitable for numeric encoding

# create a lookup table
lookupDT = data.table(oldNumber = c("None", "One", "Two", "Three", 
                                    "Four", "Five", "Six", "Seven", 
                                    "Eight", "Nine", "Ten"), 
                      newNumber = c(0:10))

# create a new column 'delinq_2yrs' with numeric conversions
lc[lookupDT, on=.('No..Delinquencies.In.Last.2.Years' = oldNumber), 
   delinq_2yrs := newNumber]

# check the conversion worked
table(lc$delinq_2yrs)
table(lc$'No..Delinquencies.In.Last.2.Years')

# create a new column 'pub_rec' with numeric conversions
lc[lookupDT, on=.('No..Adverse.Public.Records' = oldNumber), 
   pub_rec := newNumber]

# create a new column 'pub_rec_bankruptcies' with numeric conversions
lc[lookupDT, on=.('No..Of.Public.Record.Bankruptcies' = oldNumber), 
   pub_rec_bankruptcies := newNumber]

# All looks good so drop the original string-based columns
lc[, c('No..Delinquencies.In.Last.2.Years', 
       'No..Adverse.Public.Records', 
       'No..Of.Public.Record.Bankruptcies') := NULL]

##  ............................................................................
##  Engineer datetime variables                                             ####

# The 'Earliest.Credit.Line.Opened' variable is in Excel date format 
# (date origin in excel is 30 dec 1899)) create a new column 'earliest_cr_line' 
# with a date formatted from the original excel date
excel.date.to.date <- function(x){as.Date(x, origin = "1899-12-30")}
lc[, earliest_cr_line := lapply(.SD, excel.date.to.date), 
   .SDcols = 'Earliest.Credit.Line.Opened']

# the date format is not useful for glm in this format, so engineer 
# a 'credit_age' feature to represent 'months ago'
# for reproducibility use 2018-01-01 as the current date instead of Sys.Date()
# NOTE: in practice, the 'credit_age' metric would be determined at time of loan
# consideration not an unknown no of months later.
lc$credit_age <- as.numeric(difftime("2018-01-01", lc$earliest_cr_line, 
                                     units = "days")) / 30

# Drop the two now-redundant columns with earliest credit line data
lc[, c('Earliest.Credit.Line.Opened', 'earliest_cr_line') := NULL]

##  ............................................................................
##  Categorize numeric variables                                            ####

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Categorize loan application description                                 ####

# NLP would likely yield useful insights into this data, 
# but here we can generate some basic categories of length
hist(lc$Loan.Application.Description, breaks = 100) 

# large # of applicants enter no description, 1000+ words, or a single word, 
# so these will require their own categories. The others we can split into 
# categories according to length
lc$descr_cat <- "Undetermined" # initialize
lc[Loan.Application.Description == 0, descr_cat := "Blank"]
lc[Loan.Application.Description == 1, descr_cat := "Word"] # W for 1-word
lc[Loan.Application.Description == 1000, descr_cat := "LS"] #LS for life-story
lc[Loan.Application.Description >= 2 & Loan.Application.Description <= 15, 
   descr_cat := "Short"] #S for short / one-sentence
lc[Loan.Application.Description >= 16 & Loan.Application.Description <= 350, 
   descr_cat := "Medium"] #M for medium
lc[Loan.Application.Description >= 351 & Loan.Application.Description <= 999, 
   descr_cat := "Long"] #L for long

# drop original
lc$Loan.Application.Description <- NULL

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Binarize variables                                                      ####

# the delinq_2yrs can be binarized for simplicity
lc[delinq_2yrs > 0, delinq_2yrs := 1] # simple yes or no for two year delinqs

# binarize home ownership
lc$Home.Owner <- 0
lc[Home.Ownership == "MORTGAGE" | Home.Ownership == "OWN", 
   Home.Owner := 1] # simple yes or no for home ownership
lc$Home.Ownership <- NULL # drop original

# binarize bankruptcies (tiny number >1)
lc$bankruptcies <- 0
lc[pub_rec_bankruptcies >= 1, bankruptcies := 1]
lc$pub_rec_bankruptcies <- NULL # drop original

# binarize derogatory public records
lc$derog_pr <- 0
lc[pub_rec >= 1, derog_pr := 1]
lc$pub_rec <- NULL # drop original

# annual income simplify (thresholds defined by entropy / dt)
lc$salary <- "Unknown" # initialize
lc[Annual.Income <= 36000, salary := "Low"]
lc[Annual.Income > 36000 & Annual.Income <= 76000, salary := "Mid"]
lc[Annual.Income > 76000, salary := "High"]
lc$Annual.Income <- NULL # drop original

# address states -- some states have low sample size
# to avoid generalization from small sample sizes, these states are
# grouped under 'Other'
state.counts <- data.table(table(lc$Address.State))
colnames(state.counts) <- c('State', 'Count')
# use a cutoff of n > 75 for now, revisit later
lc[Address.State %in% state.counts[Count < 75, State], Address.State := 'Other']

# identify a State to use as the reference state for log reg
prop.table(table(train$Address.State, train$bClass),1)

# Pick KS (Kansas as the reference level as defaults approx 50/50)
lc$Address.State <- as.factor(lc$Address.State)
lc$Address.State <- relevel(lc$Address.State, ref = "KS")
levels(lc$Address.State) # KS is now first / reference


#   ____________________________________________________________________________
#   Split data for model training                                           ####

# prepare train/test split (70/30) using stratified random sampling
set.seed(95)
sample <- createDataPartition(lc$bClass, p = .7, list = FALSE)
train.imba <- lc[sample, ]
test  <- data.table(lc[-sample, ])

# examine class balance
prop.table(table(train.imba$bClass))
prop.table(table(test$bClass))
prop.table(table(lc$bClass))

# Note: class imbalance (0.18 / 0.82) is likely to lead to poor modelling
# of the minority (uncreditworthy) class, and an increase in type I errors
# (i.e. incorrectly predicting uncreditworthy individuals as creditworthy)

##  ............................................................................
##  Oversampling to address class imbalance                                 ####

# oversampling is performed on train data to prevent contamination of unseen

train <- data.table(ovun.sample(bClass ~ ., data = train.imba, 
                                p = 0.5, seed = 95, method = "over")$data)


# quick checks to ensure that: -
# the training data set has a 50/50 class balance
# the train / test split is 70/30
# the test data is unseen (i.e. unique entries absent in train set)

dim(train)
dim(train.imba)
table(train$bClass)

# examine unique cases
#aggregate(id ~ bClass, train, function(x) length(unique(x)))
#aggregate(id ~ bClass, train.imba, function(x) length(unique(x)))
#aggregate(id ~ bClass, test, function(x) length(unique(x)))

# all test data is unseen
sum(test$id %in% train$id)


#   ____________________________________________________________________________
#   Model training                                                          ####


##  ............................................................................
##  Full logistic regression model                                          ####

#model <- glm(bClass ~  . -id, family=binomial, data = train) # drop id

# count the number of non-redundant models
#model <- glmulti(bClass ~  . -id, family=binomial, method = "d", 
#                 data = train, level = 1) # drop id


##  ............................................................................
##  Automated Model Selection with glmulti                                  ####

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Fit all non-redundant candidate models with glmulti                     ####

#m.model <- glmulti(bClass ~  . -id, family=binomial, 
#                    data = train, level = 1, method = "h") # drop id

### . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ..
### Fit models with Genetic Algorithm                                       ####

#g.model <- glmulti(bClass ~  . -id, family=binomial, 
#                    data = train, level = 1, method = "g") # drop id

# After 1200 generations:
# this is the optimal model after examining 1200 generations (Genetic Algo)
# on the balanced classes (via oversampling) data
model <- glm(bClass ~ 1 + Loan.Amount + Loan.Term + salary + 
               Loan.Purpose + Address.State + Debt.To.Income.Ratio + 
               FICO.Credit.Score + No..Inquiries.In.Last.6.Months + 
               Months.Since.Last.Delinquency + Use.Of.Credit.Line +
               Total.Number.Of.Credit.Lines + delinq_2yrs + derog_pr +
               bankruptcies + credit_age + descr_cat + Home.Owner - id,
             family = binomial, data = train)


summary(model)


# retrieve model coefficients

model.tidy <- data.table(tidy(model))
# calculate odd ratio with 95% CI 
model.tidy$odd.ratio <- exp(model.tidy$estimate)
model.tidy$OR.upperCI <- exp(model.tidy$estimate + 
                               qnorm(0.975) * model.tidy$std.error)
model.tidy$OR.lowerCI <- exp(model.tidy$estimate - 
                               qnorm(0.975) * model.tidy$std.error)
#model.tidy

#   ____________________________________________________________________________
#   Evaluate model                                                          ####

# anova(model, test = "Chisq")


##  ............................................................................
##  Predict classes on test data set                                        ####

y_pred <- plogis(predict(model, newdata = test))

# determine the optimal cutoff for binarization of prediction values
# optimize for misclassificationerrors by default
optCutOff <- optimalCutoff(test$bClass, y_pred, 
                           optimiseFor = "Both")[1]
optCutOff


#y_pred_bClass <- ifelse(y_pred > 0.7, 1, 0) # binarize
y_pred_bClass <- ifelse(y_pred > optCutOff, 1, 0) # binarize



##  ............................................................................
##  Evaluation metrics and plots                                            ####

#calculate accuracy of model
mcerror <- mean(y_pred_bClass != test$bClass) # misclassification error
print(paste('Accuracy: ', 1 - mcerror))

Gini(y_pred, test$bClass)
youdensIndex(actuals = test$bClass, predictedScores = y_pred_bClass)
misClassError(actuals = test$bClass, predictedScores = y_pred_bClass)
kappaCohen(actuals = test$bClass, predictedScores = y_pred_bClass)
Concordance(actuals = test$bClass, predictedScores = y_pred)

  
#confusion matrix

require(caret)
CrossTable(test$bClass, y_pred_bClass, prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, dnn = c('actual', 'predicted'))
caret::confusionMatrix(y_pred_bClass, test$bClass, positive = "1") # better


# ROC curve with AUC
rocCurve = roc(response = test$bClass, 
               predictor = y_pred)

auc_curve = auc(rocCurve)

plot(rocCurve, legacy.axes = TRUE, print.auc = TRUE, col="red", main="ROC")

# KS stat
ks_stat(actuals = test$bClass, predictedScores = y_pred_bClass)
ks_plot(actuals = test$bClass, predictedScores = y_pred_bClass)

ks_stat(actuals = test$bClass, predictedScores = y_pred)
ks_plot(actuals = test$bClass, predictedScores = y_pred)

# hmeasure (prof Hand)
misclassCounts(predicted.class = y_pred_bClass, true.class = test$bClass)
summary(HMeasure(test$bClass, y_pred, threshold = optCutOff))
plotROC(HMeasure(test$bClass, y_pred, threshold = optCutOff))


#   ____________________________________________________________________________
#   Plots                                                                   ####


# Word counts for loan description
table(lc$descr_cat)
prop.table(table(lc$descr_cat, lc$bClass), 1)
plot(prop.table(table(lc$descr_cat, lc$bClass), 1))



##  ............................................................................
##  Coefficients / Odd-ratios                                               ####

# Plot odd.ratio for each coefficient in the model
coefficient.subset <- model.tidy[term != "(Intercept)"]

# plot 
ggplot(data = coefficient.subset, aes(x = odd.ratio, y = term))+
  geom_errorbarh(aes(xmax = OR.upperCI, xmin = OR.lowerCI), 
                 size = .5, color = "gray50") +
  geom_point(size = 3, colour = "orange") +
  geom_vline(aes(xintercept = 1), size = .25, linetype = "dashed") + 
  theme_bw() +
  theme(panel.grid.minor = element_blank()) +
  ylab("Predictor") +
  xlab("Odds Ratio (log scale)") +
  coord_trans(x = "log10") +
  #ggtitle("Creditworthiness of Loan Applicants")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Plot non-geographic odds ratios
coefficient.subset <- model.tidy[term != "(Intercept)" & 
                                   !(term %like% "Address.State")]

ggplot(data = coefficient.subset, aes(x = odd.ratio, y = term))+
  geom_errorbarh(aes(xmax = OR.upperCI, xmin = OR.lowerCI), 
                 size = .5, color = "gray50") +
  geom_point(size = 3, colour = "orange") +
  geom_vline(aes(xintercept = 1), size = .25, linetype = "dashed") + 
  theme_bw() +
  theme(panel.grid.minor = element_blank()) +
  ylab("Predictor") +
  xlab("Odds Ratio (log scale)") +
  coord_trans(x = "log10") +
  #ggtitle("Creditworthiness of Loan Applicants")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
# Plot of word length descrption

##  ............................................................................
##  Odd ratios by US State                                                  ####

# retrieve model properties relating to State
state.odds <- model.tidy[term %like% "Address.State",]
state.odds$state <- sapply(strsplit(state.odds$term, split='Address.State', 
                                    fixed=TRUE), 
                           function(x) (x[2])) # retrieve state code

# keep only the stats needed for the plot
state.odds <- state.odds[, .(state, odd.ratio)]

# prepare additional columns to plot a map

# Define the text label shown when cursor hovers over the State
state.odds$hover <- with(state.odds, paste(state, '<br>', "Odd.Ratio:", 
                                           odd.ratio))


# convert odd.ratio < 1 to fc to linearize for plotting
state.odds$odd.ratio.fc <- state.odds$odd.ratio
state.odds[odd.ratio < 1, odd.ratio.fc := -(1/odd.ratio)] 

# Create a Chloropleth map with plot.ly

# give state boundaries a black border
l <- list(color = toRGB("black"), width = 1)

# specify some map projection/options
g <- list(
  scope = 'usa',
  projection = list(type = 'albers usa'),
  showlakes = TRUE,
  lakecolor = toRGB('white')
)

p <- plot_geo(state.odds[state != "Other",], locationmode = 'USA-states') %>%
  add_trace(
    z = ~odd.ratio.fc, text = ~hover, locations = ~state,
    color = ~odd.ratio.fc, 
    colors = "RdBu",
    zmin = -2,
    zmax = 2,
    marker = list(line = l)
  ) %>%
  colorbar(title = "Relative Risk") %>%
  layout(
    title = 'LendingClub - Lending Risk by State \n (Logistic Regression Model)',
    geo = g
  )

Sys.setenv("plotly_username"="Combo9")
Sys.setenv("plotly_api_key"="G7Tk9Q3kgRuRq0NWjwv4")

chart_link = api_create(p, filename="lgmap")
chart_link

#   ____________________________________________________________________________
#   Scrapbook / Junk                                                        ####

# The following code is scrap that may come in handy in a future analysis.
# some ideas include changepoint analysis and entropy calculations for
# categorization of numeric variables, and the use of ML techniques to
# inform selection of variables for logistic regression and to identify
# conjunctive/disjunctive features that may inform introduction of
# interaction terms into the model

# Classification Tree with rpart

require(rattle)
model <- rpart(as.factor(bClass) ~ . -id, data = lc,
                      control = rpart.control(minbucket = 50, 
                                              minsplit = 50, cp = 0.003))

fancyRpartPlot(model)

# entropy / changepoint scrapbook

require(entropy)
require(changepoint)

cdplot(as.factor(bClass) ~ FICO.Credit.Score, data = train, ylevels = 2:1)

cdplot(as.factor(bClass) ~ as.factor(salary), data = train, ylevels = 2:1)

mvalue = cpt.mean(e.df$ent, method="PELT") #mean changepoints using PELT
mvalue = cpt.mean(e.df$ent, method = "AMOC", Q = 1) #mean changepoints
mvalue
cpts(mvalue)
plot(mvalue)

# conditional density plots to reveal relationship between numeric 
# predictor variables and output.  Can inform categorization decisions
# and cutoffs.

cdplot(as.factor(bClass) ~ Annual.Income, data = train, ylevels = 2:1, 
       bw = "nrd0")

cdplot(as.factor(bClass) ~ Annual.Income, data = lc, ylevels = 2:1, 
       bw = 5, yaxlabels = "n")
axis(4)

cdplot(as.factor(bClass) ~ FICO.Credit.Score, data = lc, ylevels = 2:1, bw = 2)


cdplot(as.factor(bClass) ~ Loan.Application.Description, data = train, 
       ylevels = 2:1, bw = 1)
cdplot(as.factor(bClass) ~ Loan.Application.Description, 
       data = train[Loan.Application.Description <= 100,], ylevels = 2:1, bw = 1)
cdplot(as.factor(bClass) ~ Annual.Income, data = train[Annual.Income <= 200000,], 
       ylevels = 2:1)

cdplot(as.factor(bClass) ~ Loan.Amount, data = train, ylevels = 2:1)
cdplot(as.factor(bClass) ~ FICO.Credit.Score, data = train, ylevels = 2:1)


cdplot(as.factor(bClass) ~ FICO.Credit.Score, data = train, ylevels = 2:1)
cdplot(as.factor(bClass) ~ FICO.Credit.Score, data = train, ylevels = 2:1, bw = 2)


# plots for report
ggplot(data=lc, aes(lc$Loan.Application.Description)) + geom_histogram()+theme_bw()

ggplot(data=data.table(y_pred), aes(y_pred)) + geom_histogram()+theme_bw()



#### EOF
