#!/usr/bin/env python

"""
model tests
"""

import unittest
from modelling import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
    
    def test_01_train(self):
        """
        test the train functionality
        """
    
        ## train the model
        model_train(verbose=False)
        
        prefix = "test" if DEV else "prod"
        models = [f for f in os.listdir(MODEL_DIR) if re.search(prefix,f)]
        self.assertEqual(len(models),11)
        
    def test_02_load(self):
        """
        test the train functionality
        """
        
        ## load the model
        models = model_load(verbose=False)
        
        for tag, model in models.items():
            self.assertTrue("predict" in dir(model))
            self.assertTrue("fit" in dir(model))
        
    def test_03_predict(self):
        """
        test the predict function input
        """
    
        ## query inputs
        query = "2018", "1", "5", "total"
        
        ## load model first
        result = model_predict(year=query[0], month=query[1], day=query[2], country=query[3], verbose=False)
        y_pred = result["y_pred"]
        self.assertTrue(y_pred.dtype==np.float64)
            
    def test_04_predict(self):
        """
        test the predict function accuracy
        """
    
        ## example predict
        example_queries = [("2018", "1", "5", "total"),
                           ("2019", "2", "5", "eire"),
                           ("2018", "12", "5", "france")]
        
        for query in example_queries:
            result = model_predict(year=query[0], month=query[1], day=query[2], country=query[3], verbose=False)
            y_pred = result["y_pred"]
            self.assertTrue(y_pred.dtype==np.float64)
            
## run the tests
if __name__ == "__main__":
    unittest.main()
