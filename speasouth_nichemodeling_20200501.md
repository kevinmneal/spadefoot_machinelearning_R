Predicting spadefoot toad environmental niche using random forest and
xgboost in R
================
Kevin Neal
November 13, 2019

*last updated May 7, 2020*

## What is Environmental Niche Modeling (i.e Species Distribution Modeling)?

  - predict habitat suitability or likelihood of occupancy for a species
    based on known occurrences and environmental data
  - useful for:
      - predicting new localities
      - understanding biological niche space and environmental variables
        that contribute to creating this space
      - predicting past and future distributions
  - Personal usage: compared model-derived niche space to provide
    evidence for splitting a species of spadefoot toad into two (Neal et
    al. 2018)

![Species Distribution Modeling](http://i.imgur.com/EHO6XOf.png?1)

![Spea hammondii (western spadefoot
toad)](http://i.imgur.com/PsRfF6F.jpg?1)

*Response variable: binary presence (1) or absence (0)*

*Predictor environmental variables: * Various sources derived from
precipitation and temperature, as well as soil and topography

### Data wrangling: Filtering presence points

  - First, want to filter points that may lie outside the study area or
    have erroneous coordinates (e.g. the point is in the ocean)
  - Want a balanced response variable - while spatial samples may
    accurately reflect species density, more likely it reflects search
    effort, so we want to filter the points
  - This can be distance/radius-based or based on other aspects of the
    localities if we have the metadata
  - Here we’ll sample points using a grid, with one sample point per
    grid cell

<!-- end list -->

``` r
# load points and rasters, filter points, make background/pseudoabsence points
# points from GBIF, Bison, iNaturalist, and Shaffer lab collections
# coords last updated in 2017
present <- read.csv("speahammondii_combined_presence_points.csv")
head(present)
```

    ##            Species Longitude Latitude
    ## 1 Spea_hammondii_S -115.6091 30.12889
    ## 2 Spea_hammondii_S -115.7599 30.19903
    ## 3 Spea_hammondii_S -115.9500 30.76833
    ## 4 Spea_hammondii_S -115.9506 30.76841
    ## 5 Spea_hammondii_S -115.7333 30.96667
    ## 6 Spea_hammondii_S -116.2017 31.06722

``` r
#dim(present)
#dim(unique(present))
```

``` r
socalstack <- stack(list.files(path="./socal_rasters/", pattern="asc$", full.names = TRUE))
socalfilenames <- list.files(path="./socal_rasters/", pattern="asc$", full.names = FALSE)
socalnames <- gsub(pattern="_.*asc", replacement="", socalfilenames)
names(socalstack) <- socalnames

pres.sub <- crop(SpatialPoints(present[,2:3]), socalstack[[1]])

# subsample presence points by a lower-resolution grid

#pres.thin <- gridSample(pres.sub, r=socalstack[[1]])
pres.thin <- gridSample(pres.sub, r=aggregate(socalstack[[1]], fact=4), n=1)

plot(socalstack[[1]], col=viridis(20), main="Spatially subsampled toad presences")
points(pres.sub, pch=4, col="red")
points(pres.thin, pch=20, col="black")
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

### Generating pseudoabsences

  - Don’t have “true” absences, but we can generate a sample of points
    that cover most of the area in question, but with a buffer around
    the presence points that is unsampled
  - Spatially filter as above, using a grid

<!-- end list -->

``` r
# take a random sample of the background, excluding cells that contain presence points, by masking the raster by the SpatialPoints object

# want a lot of pseudoabsences to get a full representation of the environmental conditions
# subsample by lower-resolution grid
set.seed(99)
absent <- randomPoints(mask=mask(aggregate(socalstack[[1]], fact=4), 
                       buffer(pres.sub, #buffer(SpatialPoints(present[,2:3]), 
                       width=6000), inverse=T), 
                       n=nrow(pres.thin), #consider making imbalanced to allow for wider coverage of the space of environmental variable combinations in the region
                       p=SpatialPoints(present[,2:3]), 
                       excludep=TRUE)
#nrow(gridSample(absent, socalstack[[1]]))

abs.thin <- gridSample(absent, r=socalstack[[1]])
```

``` r
plot(absent, pch=4, col="red")
#points(abs.thin, col="blue")
points(pres.thin, col="green")
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
#points
```

``` r
# randomly select from the absences equal to presences to have balanced set
## not needed if n=length(pres.thin) in generating random points
#set.seed(1)
#rows_to_sample <- sample(1:nrow(), nrow(pres.thin))
#abs.thin <- abs.thin[rows_to_sample,] # sample random rows without replacement
dim(abs.thin)
```

    ## [1] 133   2

``` r
dim(pres.thin)
```

    ## [1] 133   2

### Combine presence and absence points into a single dataframe

``` r
# make dataframes specifying point type (present=1, absent=0) 
presabs <- dplyr::union(
  data.frame(pa="present", Longitude=pres.thin[,1], Latitude=pres.thin[,2]), 
  data.frame(pa="absent", Longitude=abs.thin[,1], Latitude=abs.thin[,2])
  )
```

    ## Warning: Column `pa` joining factors with different levels, coercing to
    ## character vector

``` r
presabs[,1] <- as.factor(presabs[,1])
presabs[,1] <- factor(presabs[,1], levels=rev(levels(presabs[,1])))
#head(presabs)
```

### Extract values (predictor variables/features) from environmental layers at each presence and absence point

``` r
# drop features related to urbanization - want the "natural" niche of the species
socalstack <- dropLayer(socalstack, c("canopy", "impervious"))

# extract environmental data at each point
presabs.envdata <- raster::extract(socalstack, presabs[,2:3])
presabs.data <- bind_cols(data.frame(presabs), data.frame(presabs.envdata))

# remove rows with missing data (i.e. points that don't fall on the map)
presabs.data <- presabs.data[complete.cases(presabs.data),]

plot(presabs.data[presabs.data$pa=="present",c(2,3)], col="blue", pch=20)
points(presabs.data[presabs.data$pa=="absent",c(2,3)], col="red", pch=20)
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

### Remove highly correlated features

  - Inclusion of highly correlated features can bias the models and
    produce misleading feature importances
  - Multiple ways to select features; I’ll use findCorrelation in caret
    to do pairwise removal of variables with spearman’s rho above 0.8

<!-- end list -->

``` r
X <- presabs.data[,-c(1:3)]
y <- presabs.data[,1]
ycoords <- presabs.data[,c(2:3)]

# use caret::findCorrelation to remove correlated variables
# could also use boruta or another mutual information method
library(corrplot)
```

    ## corrplot 0.84 loaded

``` r
library(gplots)
```

    ## 
    ## Attaching package: 'gplots'

    ## The following object is masked from 'package:stats':
    ## 
    ##     lowess

``` r
heatmap.2(abs(cor(X, method="spearman")), symm=T, col=magma(10), trace="none")
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
remove.vars <- findCorrelation(cor(X, method="spearman"), cutoff=0.8, names=TRUE, verbose=F, exact=TRUE)
X.sel <- X[, !names(X) %in% remove.vars] # remove the correlated variables
```

``` r
#pairs(presabs.data.sel)
names(X.sel)
```

    ##  [1] "annualPET"                "aridityIndexThornthwaite"
    ##  [3] "bio03"                    "bio05"                   
    ##  [5] "bio13"                    "bio14"                   
    ##  [7] "bio15"                    "bulkdensity5cm"          
    ##  [9] "claycontent5cm"           "depthtobedrockrhorizon"  
    ## [11] "embergerQ"                "minTempWarmest"          
    ## [13] "PETWettestQuarter"        "siltcontent5cm"          
    ## [15] "toporuggedness"

``` r
### findCorrelation is pretty crude. Try Boruta (permuted random forest feature importance)

library(Boruta)
```

    ## Warning: package 'Boruta' was built under R version 3.6.3

    ## Loading required package: ranger

    ## 
    ## Attaching package: 'ranger'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     importance

``` r
boruta_results <- Boruta(x=X.sel,
                       y=y,
                         doTrace=1,
                       getImp=getImpExtraRaw)
```

    ## After 11 iterations, +1.9 secs:

    ##  confirmed 14 attributes: annualPET, aridityIndexThornthwaite, bio03, bio05, bio13 and 9 more;

    ##  still have 1 attribute left.

    ## After 44 iterations, +6.8 secs:

    ##  confirmed 1 attribute: depthtobedrockrhorizon;

    ##  no more attributes left.

``` r
boruta_keep = names(boruta_results$finalDecision[boruta_results$finalDecision=="Confirmed"])
boruta_keep
```

    ##  [1] "annualPET"                "aridityIndexThornthwaite"
    ##  [3] "bio03"                    "bio05"                   
    ##  [5] "bio13"                    "bio14"                   
    ##  [7] "bio15"                    "bulkdensity5cm"          
    ##  [9] "claycontent5cm"           "depthtobedrockrhorizon"  
    ## [11] "embergerQ"                "minTempWarmest"          
    ## [13] "PETWettestQuarter"        "siltcontent5cm"          
    ## [15] "toporuggedness"

``` r
X.sel <- X.sel[, names(X.sel) %in% boruta_keep]
```

``` r
# blue = shadow (permuted) variable
# green = higher importance than maximum shadow (permuted) variable
# red = below max shadow importance and should be dropped
plot(boruta_results, las=2)
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
# remove variables from the raster stack
#socalstack.sub <- subset(socalstack, names(socalstack)[!names(socalstack) %in% remove.vars])
socalstack_sub <- subset(socalstack, boruta_keep) #setdiff(names(socalstack), remove.vars))
names(socalstack_sub)
```

    ##  [1] "annualPET"                "aridityIndexThornthwaite"
    ##  [3] "bio03"                    "bio05"                   
    ##  [5] "bio13"                    "bio14"                   
    ##  [7] "bio15"                    "bulkdensity5cm"          
    ##  [9] "claycontent5cm"           "depthtobedrockrhorizon"  
    ## [11] "embergerQ"                "minTempWarmest"          
    ## [13] "PETWettestQuarter"        "siltcontent5cm"          
    ## [15] "toporuggedness"

``` r
# this is the complete dataframe that will be the input in the models
```

### Exploratory data analysis

  - Qualitatively, do any features show clear discrimination between our
    two response classes (presence and absence)?

<!-- end list -->

``` r
# examine distribution of feature values by class
featurePlot(x = X.sel, 
            y = y, # must be factor 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")),
            auto.key=TRUE)
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

### Modelling using caret

  - Decision trees have proven to be useful for classification; I’ll try
    two ensemble tree methods: random forest (using both gini and
    extratrees) and xgboost and compare the results, and use grid search
    with cross-validation to tune hyperparameters

<!-- end list -->

``` r
# run models using caret, starting with xgboost, then ranger/rf


xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 5, 
  repeats = 3,
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE,
  savePredictions = "final",
  #summaryFunction = twoClassSummary #use with metric="ROC" in train(); or use prSummary with AUC
  summaryFunction = prSummary
  #sampling="down"
)
```

``` r
#trainall <- allpts.thin.data
#trainall[,1] <- make.names(trainall[,1])


modelLookup("xgbTree")
```

    ##     model        parameter                          label forReg forClass
    ## 1 xgbTree          nrounds          # Boosting Iterations   TRUE     TRUE
    ## 2 xgbTree        max_depth                 Max Tree Depth   TRUE     TRUE
    ## 3 xgbTree              eta                      Shrinkage   TRUE     TRUE
    ## 4 xgbTree            gamma         Minimum Loss Reduction   TRUE     TRUE
    ## 5 xgbTree colsample_bytree     Subsample Ratio of Columns   TRUE     TRUE
    ## 6 xgbTree min_child_weight Minimum Sum of Instance Weight   TRUE     TRUE
    ## 7 xgbTree        subsample           Subsample Percentage   TRUE     TRUE
    ##   probModel
    ## 1      TRUE
    ## 2      TRUE
    ## 3      TRUE
    ## 4      TRUE
    ## 5      TRUE
    ## 6      TRUE
    ## 7      TRUE

``` r
xgb.grid.large <- expand.grid(nrounds = c(100, 500), # default
                         eta = c(0.1, 0.5), # default
                         max_depth = c(3,5,7,9,11),
                         gamma=0, # default
                         colsample_bytree=1, # default
                         min_child_weight=1, # default
                         subsample=1 # default
                         )


set.seed(99)
xgb_caret <- train(x=X.sel,
                   y=y,
                   trControl=xgb_trcontrol,
                   method="xgbTree",
                   tuneGrid=xgb.grid.large,
                   #metric="ROC", #"AUC"
                   metric="AUC", # good for imbalanced problems
                   importance="permutation")

xgb_caret
```

    ## eXtreme Gradient Boosting 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 189, 189, 190, 190, 190, 189, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   eta  max_depth  nrounds  AUC        Precision  Recall     F        
    ##   0.1   3         100      0.7763398  0.7461583  0.7641026  0.7525200
    ##   0.1   3         500      0.7691969  0.7380203  0.7615385  0.7475035
    ##   0.1   5         100      0.7719995  0.7511812  0.7743590  0.7603145
    ##   0.1   5         500      0.7746443  0.7441766  0.7769231  0.7581310
    ##   0.1   7         100      0.7775282  0.7536105  0.7743590  0.7608609
    ##   0.1   7         500      0.7777739  0.7588136  0.7666667  0.7599926
    ##   0.1   9         100      0.7777575  0.7509527  0.7769231  0.7609979
    ##   0.1   9         500      0.7753455  0.7467552  0.7692308  0.7553962
    ##   0.1  11         100      0.7718846  0.7472103  0.7794872  0.7608983
    ##   0.1  11         500      0.7724037  0.7380513  0.7666667  0.7497359
    ##   0.5   3         100      0.7703545  0.7226122  0.7512821  0.7345117
    ##   0.5   3         500      0.7642056  0.7282703  0.7615385  0.7419486
    ##   0.5   5         100      0.7786043  0.7296875  0.7692308  0.7469702
    ##   0.5   5         500      0.7781908  0.7350216  0.7717949  0.7506677
    ##   0.5   7         100      0.7764998  0.7461842  0.7769231  0.7586501
    ##   0.5   7         500      0.7757290  0.7253521  0.7769231  0.7481820
    ##   0.5   9         100      0.7775669  0.7356821  0.7589744  0.7449589
    ##   0.5   9         500      0.7767521  0.7336964  0.7717949  0.7491779
    ##   0.5  11         100      0.7773090  0.7350782  0.7564103  0.7432718
    ##   0.5  11         500      0.7765384  0.7354641  0.7692308  0.7485112
    ## 
    ## Tuning parameter 'gamma' was held constant at a value of 0
    ##  1
    ## Tuning parameter 'min_child_weight' was held constant at a value of
    ##  1
    ## Tuning parameter 'subsample' was held constant at a value of 1
    ## AUC was used to select the optimal model using the largest value.
    ## The final values used for the model were nrounds = 100, max_depth = 5,
    ##  eta = 0.5, gamma = 0, colsample_bytree = 1, min_child_weight = 1
    ##  and subsample = 1.

``` r
ggplot(xgb_caret)
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
# library(rasterVis)
# levelplot(xgb_predtest_prob, par.settings=rasterTheme(parula(20)))
```

``` r
plot(varImp(xgb_caret, scale=F)) # permutation importance
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
#modelLookup("ranger")

p = length(names(socalstack_sub))
mtrys = c(
  floor(sqrt(p)),
  floor(p*0.33),
  floor(p*0.5),
  p
)

rf.grid <- expand.grid(
  mtry = mtrys, #c(3,4,5,11),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(2,5,10)
)


set.seed(99)
ranger_caret <- train(x=X.sel,
                   num.trees=500,
                   y=y,
                   trControl=xgb_trcontrol,
                   method="ranger",
                   tuneGrid=rf.grid,
                   #metric="ROC", #"AUC",
                   metric="AUC",
                   importance="permutation")

ranger_caret
```

    ## Random Forest 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 189, 189, 190, 190, 190, 189, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   min.node.size  AUC        Precision  Recall   
    ##    3    gini         2             0.8057957  0.7781564  0.7692308
    ##    3    gini         5             0.8095058  0.7806395  0.7692308
    ##    3    gini        10             0.8094645  0.7751306  0.7461538
    ##    3    extratrees   2             0.8111774  0.7522368  0.8051282
    ##    3    extratrees   5             0.8075571  0.7667263  0.8076923
    ##    3    extratrees  10             0.8111248  0.7702917  0.7974359
    ##    4    gini         2             0.8021481  0.7641103  0.7641026
    ##    4    gini         5             0.8032412  0.7685866  0.7666667
    ##    4    gini        10             0.8069094  0.7689187  0.7435897
    ##    4    extratrees   2             0.8118849  0.7629189  0.8000000
    ##    4    extratrees   5             0.8086911  0.7648278  0.7974359
    ##    4    extratrees  10             0.8068295  0.7699280  0.7923077
    ##    7    gini         2             0.8012131  0.7687578  0.7589744
    ##    7    gini         5             0.8027285  0.7742789  0.7461538
    ##    7    gini        10             0.8013189  0.7767765  0.7461538
    ##    7    extratrees   2             0.8047099  0.7559935  0.8025641
    ##    7    extratrees   5             0.8066796  0.7677002  0.8000000
    ##    7    extratrees  10             0.8055008  0.7740327  0.7846154
    ##   15    gini         2             0.7901325  0.7591479  0.7410256
    ##   15    gini         5             0.7930368  0.7704770  0.7615385
    ##   15    gini        10             0.7973522  0.7729169  0.7538462
    ##   15    extratrees   2             0.8026432  0.7591484  0.7820513
    ##   15    extratrees   5             0.8045353  0.7713595  0.7948718
    ##   15    extratrees  10             0.8031933  0.7797810  0.7769231
    ##   F        
    ##   0.7696257
    ##   0.7715470
    ##   0.7568761
    ##   0.7759046
    ##   0.7843636
    ##   0.7814165
    ##   0.7607861
    ##   0.7643994
    ##   0.7511080
    ##   0.7787970
    ##   0.7785160
    ##   0.7783252
    ##   0.7615559
    ##   0.7553898
    ##   0.7576480
    ##   0.7764951
    ##   0.7807372
    ##   0.7758861
    ##   0.7472396
    ##   0.7635959
    ##   0.7607015
    ##   0.7678372
    ##   0.7803918
    ##   0.7744365
    ## 
    ## AUC was used to select the optimal model using the largest value.
    ## The final values used for the model were mtry = 4, splitrule =
    ##  extratrees and min.node.size = 2.

``` r
ggplot(ranger_caret)
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

``` r
#
```

``` r
plot(varImp(ranger_caret, scale=F))
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

## Get mean AUC, precision, recall, F1 for cross-validation folds

### ranger:

``` r
getTrainPerf(ranger_caret)
```

    ##    TrainAUC TrainPrecision TrainRecall   TrainF method
    ## 1 0.8118849      0.7629189         0.8 0.778797 ranger

``` r
# equivalent to:
# apply(ranger_caret$resample[,-5], 2, "mean")

confusionMatrix(ranger_caret)
```

    ## Cross-Validated (5 fold, repeated 3 times) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction present absent
    ##    present    43.9   13.9
    ##    absent     11.0   31.2
    ##                             
    ##  Accuracy (average) : 0.7511

``` r
#twoClassSummary(data=ranger_caret$pred, lev=levels(ranger_caret$pred$obs))
#prSummary(data=ranger_caret$pred, lev=levels(ranger_caret$pred$obs))
```

### xgboost:

``` r
getTrainPerf(xgb_caret)
```

    ##    TrainAUC TrainPrecision TrainRecall    TrainF  method
    ## 1 0.7786043      0.7296875   0.7692308 0.7469702 xgbTree

``` r
confusionMatrix(xgb_caret)
```

    ## Cross-Validated (5 fold, repeated 3 times) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction present absent
    ##    present    42.2   15.8
    ##    absent     12.7   29.4
    ##                             
    ##  Accuracy (average) : 0.7159

``` r
#twoClassSummary(data=xgb_caret$pred, lev=levels(xgb_caret$pred$obs))
#prSummary(data=xgb_caret$pred, lev=levels(xgb_caret$pred$obs))
```

### boxplots of cross-validation folds for both models:

``` r
results <- resamples(list(RF=ranger_caret, XGB=xgb_caret))
# summarize the distributions
summary(results)
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: RF, XGB 
    ## Number of resamples: 15 
    ## 
    ## AUC 
    ##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## RF  0.6788524 0.7642945 0.7968090 0.8118849 0.8711055 0.9001120    0
    ## XGB 0.6740231 0.7320556 0.7774966 0.7786043 0.8154263 0.8885378    0
    ## 
    ## F 
    ##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## RF  0.7058824 0.7284211 0.7931034 0.7787970 0.8199280 0.8474576    0
    ## XGB 0.6415094 0.7269116 0.7636364 0.7469702 0.7735043 0.8196721    0
    ## 
    ## Precision 
    ##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## RF  0.6666667 0.7193750 0.7575758 0.7629189 0.8148148 0.8695652    0
    ## XGB 0.6296296 0.7089947 0.7272727 0.7296875 0.7646154 0.8076923    0
    ## 
    ## Recall 
    ##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## RF  0.6923077 0.7692308 0.8076923 0.8000000 0.8461538 0.9615385    0
    ## XGB 0.6153846 0.7307692 0.7692308 0.7692308 0.8076923 0.9615385    0

``` r
# boxplots of results
bwplot(results)
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

  - RF and XGBoost perform pretty similarly in cross-validation

## Re-train models on full dataset and do predictions on the rasters

``` r
xgb_trcontrol_fulltrain = trainControl(
  method = "none",
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE,
  classProbs = TRUE,
  savePredictions = "final",
  #summaryFunction = twoClassSummary #use with metric="ROC" in train(); or use prSummary with AUC
  summaryFunction = prSummary
  #sampling="down"
)

set.seed(99)
xgb.grid.best <- expand.grid(nrounds = xgb_caret$bestTune$nrounds,
                         eta = xgb_caret$bestTune$eta,
                         max_depth = xgb_caret$bestTune$max_depth,
                         gamma=xgb_caret$bestTune$gamma, # default
                         colsample_bytree=xgb_caret$bestTune$colsample_bytree, # default
                         min_child_weight=xgb_caret$bestTune$min_child_weight, # default
                         subsample=xgb_caret$bestTune$subsample # default
                         )

xgb_caret_best <- train(x=X.sel,
                   y=y,
                   trControl=xgb_trcontrol_fulltrain,
                   method="xgbTree",
                   tuneGrid=xgb.grid.best,
                   #metric="ROC", #"AUC"
                   metric="AUC",
                   importance="permutation")

xgb_predtest_prob <- raster::predict(socalstack_sub, xgb_caret_best, type="prob") # shows class assignment by default; use type="prob" to show class prob
xgb_predtest_bin <- raster::predict(socalstack_sub, xgb_caret_best, type="raw")

#par(mar=c(2,2,2,1))
#plot(xgb_predtest_prob, col=magma(20), main="presence probability, XGB") 


rf.grid.best <- expand.grid(
  mtry = ranger_caret$bestTune$mtry,
  splitrule = ranger_caret$bestTune$splitrule,
  min.node.size = ranger_caret$bestTune$min.node.size
)

ranger_caret_best <- train(x=X.sel,
                   num.trees=500,
                   y=y,
                   trControl=xgb_trcontrol_fulltrain,
                   method="ranger",
                   tuneGrid=rf.grid.best,
                   #metric="ROC", #"AUC",
                   metric="AUC",
                   importance="permutation")


ranger_predtest_prob <- raster::predict(socalstack_sub, ranger_caret_best, type="prob")
ranger_predtest_bin <- raster::predict(socalstack_sub, ranger_caret_best, type="raw")

#par(mar=c(2,2,2,1))
#plot(ranger_predtest_prob, col=magma(20), main="presence probability, ranger RF")
```

### Visual comparison of predictions

``` r
#par(mfrow=c(1,1))
par(mfrow=c(2,2))
par(mar=c(2,2,2,1))
plot(xgb_predtest_prob, col=magma(20), main="presence probability, XGB")
points(ycoords, col="red")
plot(ranger_predtest_prob, col=magma(20), main="presence probability, ranger RF")
points(ycoords, col="red")

plot(2-xgb_predtest_bin, col=magma(20), main="presence (binary), XGB")
points(ycoords, col="red")
plot(2-ranger_predtest_bin, col=magma(20), main="presence (binary), ranger RF")
points(ycoords, col="red")
```

![](speasouth_nichemodeling_20200501_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

``` r
par(mfrow=c(1,1))
```

``` r
# session info
sessionInfo()
```

    ## R version 3.6.1 (2019-07-05)
    ## Platform: x86_64-w64-mingw32/x64 (64-bit)
    ## Running under: Windows 10 x64 (build 17763)
    ## 
    ## Matrix products: default
    ## 
    ## locale:
    ## [1] LC_COLLATE=English_United States.1252 
    ## [2] LC_CTYPE=English_United States.1252   
    ## [3] LC_MONETARY=English_United States.1252
    ## [4] LC_NUMERIC=C                          
    ## [5] LC_TIME=English_United States.1252    
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] Boruta_6.0.0        ranger_0.11.2       gplots_3.0.1.1     
    ##  [4] corrplot_0.84       pals_1.5            maps_3.3.0         
    ##  [7] dplyr_0.8.3         gbm_2.1.5           randomForest_4.6-14
    ## [10] dismo_1.1-4         raster_3.0-7        sp_1.3-1           
    ## [13] xgboost_0.90.0.2    caret_6.0-84        ggplot2_3.2.1      
    ## [16] lattice_0.20-38    
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] jsonlite_1.6            splines_3.6.1          
    ##  [3] foreach_1.4.7           gtools_3.8.1           
    ##  [5] prodlim_2019.10.13      shiny_1.4.0            
    ##  [7] assertthat_0.2.1        stats4_3.6.1           
    ##  [9] yaml_2.2.0              ipred_0.9-9            
    ## [11] pillar_1.4.2            glue_1.3.1             
    ## [13] MLmetrics_1.1.1         digest_0.6.22          
    ## [15] manipulateWidget_0.10.0 promises_1.1.0         
    ## [17] colorspace_1.4-1        recipes_0.1.7          
    ## [19] htmltools_0.4.0         httpuv_1.5.2           
    ## [21] Matrix_1.2-17           plyr_1.8.4             
    ## [23] timeDate_3043.102       pkgconfig_2.0.3        
    ## [25] purrr_0.3.3             xtable_1.8-4           
    ## [27] scales_1.0.0            webshot_0.5.1          
    ## [29] gdata_2.18.0            later_1.0.0            
    ## [31] gower_0.2.1             lava_1.6.6             
    ## [33] tibble_2.1.3            generics_0.0.2         
    ## [35] withr_2.1.2             ROCR_1.0-7             
    ## [37] nnet_7.3-12             lazyeval_0.2.2         
    ## [39] survival_2.44-1.1       magrittr_1.5           
    ## [41] crayon_1.3.4            mime_0.7               
    ## [43] evaluate_0.14           nlme_3.1-140           
    ## [45] MASS_7.3-51.4           class_7.3-15           
    ## [47] tools_3.6.1             data.table_1.12.6      
    ## [49] stringr_1.4.0           munsell_0.5.0          
    ## [51] e1071_1.7-2             compiler_3.6.1         
    ## [53] caTools_1.17.1.2        rlang_0.4.1            
    ## [55] grid_3.6.1              dichromat_2.0-0        
    ## [57] iterators_1.0.12        htmlwidgets_1.5.1      
    ## [59] crosstalk_1.0.0         miniUI_0.1.1.1         
    ## [61] labeling_0.3            bitops_1.0-6           
    ## [63] rmarkdown_1.16          gtable_0.3.0           
    ## [65] ModelMetrics_1.2.2      codetools_0.2-16       
    ## [67] reshape2_1.4.3          R6_2.4.0               
    ## [69] gridExtra_2.3           lubridate_1.7.4        
    ## [71] knitr_1.25              rgeos_0.5-2            
    ## [73] fastmap_1.0.1           KernSmooth_2.23-15     
    ## [75] stringi_1.4.3           Rcpp_1.0.3             
    ## [77] mapproj_1.2.6           rpart_4.1-15           
    ## [79] rgl_0.100.30            tidyselect_0.2.5       
    ## [81] xfun_0.10
