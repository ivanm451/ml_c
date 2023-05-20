#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// training data set
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

#define train_count (sizeof(train)/sizeof(train[0])) // gives amount of elements to train on

// implementation of a random float using integer return rand() c func
float rand_float(void)
{
    return (float) rand()/ (float) RAND_MAX;
}

// cost function for model error check
float cost(float w){
 // here we check if our model is correctly matching the expected values from the training data set
    float result = 0.0f; // keeps track of error in our model
    for (size_t i = 0; i < train_count; ++i){
        float x = train[i][0];
        float y = x*w;
        
        // a simple check to see how bad our model is performing
        // distance between the actual and expected 
        float d = y - train[i][1];
        result += d*d; // error is always positive because of property of squares, errors are amplified as well
        //printf("actual: %f, expected: %f\n", y, train[i][0]);
    }

    // find average of entire error 
    // if result = 0, we get perfect fit 
    // Remember, perfect fit can be bad (overfitting) good model, bad predictions
    result /= train_count;
    return result; 
}

int main()
{
    //srand(time(0)); // seeding to make our randomness better
    srand(69);
    float w = rand_float()*10.0f; // model parameter;  we implement random float for first version of code
   
    float eps = 1e-3; // smaller eps, better approximation
    float rate = 1e-3; // learning rate of model
    printf("%f\n", cost(w));
    for (size_t i = 0; i < 500; ++i){
        float dcost = (cost(w + eps) - cost(w))/eps; // distance of cost function, oonga boonga derivative; finite difference (not feasible in actual products/models)
        w -= rate*dcost;
        printf("cost function = %f, w = %f\n", cost(w), w);
    }
    printf("------------------------------------------\n");
    printf("%f\n", w); 
    return 0;
}
