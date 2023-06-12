 class Metesre():
        
        def __init__(self):
            
            
            item_ids = pd.read_parquet('/kaggle/input/etl-de-nopadding/categories_DE_full_new/kaggle/working/categories/unique.prev_items.parquet')
            self.items = item_ids['prev_items']
            self.sessions_gdf = pd.read_parquet("/kaggle/input/etl-de-nopadding/processed_DE_full_new/kaggle/working/processed_nvt/part_0.parquet")
            self.test_gdf = pd.read_parquet("/kaggle/input/etl-de-nopadding/processed_DE_full_new/kaggle/working/processed_nvt/test_0.parquet")
            self.preprocessing()
            self.buildModel()
            
            
        def preprocessing(self):
            
            
            X1 = self.sessions_gdf['prev_items-list'].tolist()
            X2 = self.sessions_gdf['title-list'].tolist()
            X3 = self.sessions_gdf['brand-list'].tolist()
            X4 = self.sessions_gdf['size-list'].tolist()
            X5 = self.sessions_gdf['model-list'].tolist()
            X6 = self.sessions_gdf['color-list'].tolist()
            X7 = self.sessions_gdf['price_log_norm-list'].tolist()
            X8 = self.sessions_gdf['relative_price_to_avg_categ_id-list'].tolist()
            
            #handles variable length session sequences
            X1 = np.array(X1, dtype='object')

            #find vocab sizes
            self.vocab_size1 = max(item for sublist in X1 for item in sublist)+1
            self.vocab_size2 = max(item for sublist in X2 for item in sublist)+1
            self.vocab_size3 = max(item for sublist in X3 for item in sublist)+1
            self.vocab_size4 = max(item for sublist in X4 for item in sublist)+1
            self.vocab_size5 = max(item for sublist in X5 for item in sublist)+1
            self.vocab_size6 = max(item for sublist in X6 for item in sublist)+1
            
            print("Vocab Sizes: \n",self.vocab_size1, self.vocab_size2, self.vocab_size3, self.vocab_size4, self.vocab_size5, self.vocab_size6)
            
            #extract next item from the X1: prev_items_list, also remove last items attributes
            X1_p = []
            X2_p = []
            X3_p = []
            X4_p = []
            X5_p = []
            X6_p = []
            X7_p = []
            X8_p = []

            y_p = []

            for i in range(len(X1)):
                X1_p.append(X1[i][:-1])
                X2_p.append(X2[i][:-1])
                X3_p.append(X3[i][:-1])
                X4_p.append(X4[i][:-1])
                X5_p.append(X5[i][:-1])
                X6_p.append(X6[i][:-1])
                X7_p.append(X5[i][:-1])
                X8_p.append(X6[i][:-1])
                y_p.append(X1[i][-1])
                
        
            X1 = X1_p
            X2 = X2_p
            X3 = X3_p
            X4 = X4_p
            X5 = X5_p
            X6 = X6_p
            X7 = X7_p
            X8 = X8_p
            y= y_p
            y = np.array(y)
            self.max_len = 10
            #padding: pre for X1 and post for all others
            X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
            X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')
            X3 = pad_sequences(X3, maxlen=self.max_len, padding='pre')
            X4 = pad_sequences(X4, maxlen=self.max_len, padding='pre')
            X5 = pad_sequences(X5, maxlen=self.max_len, padding='pre')
            X6 = pad_sequences(X6, maxlen=self.max_len, padding='pre')
            X7 = pad_sequences(X7, maxlen=self.max_len, padding='pre')
            X8 = pad_sequences(X8, maxlen=self.max_len, padding='pre')

            self.X1_train, self.X1_test, self.X2_train, self.X2_test, self.X3_train, self.X3_test, self.X4_train, \
            self.X4_test , self.X5_train, self.X5_test, self.X6_train, self.X6_test, self.X7_train, self.X7_test, self.X8_train, self.X8_test,  self.y_train, self.y_test = train_test_split(X1, X2, X3, X4, \
                                                                                              X5, X6, X7, X8, y, test_size=0.005,random_state=42, shuffle=True)
        
        def buildModel(self):
            
            
            embedding_dim = 128
            hidden_units = 512
            seq_length = self.max_len

            # Define the input layers
            input_layer1 = tf.keras.Input(shape=(seq_length,))
            input_layer2 = tf.keras.Input(shape=(seq_length,))
            input_layer3 = tf.keras.Input(shape=(seq_length,))
            input_layer4 = tf.keras.Input(shape=(seq_length,))
            input_layer5 = tf.keras.Input(shape=(seq_length,))
            input_layer6 = tf.keras.Input(shape=(seq_length,))
            input_layer7 = tf.keras.Input(shape=(seq_length,))
            input_layer8 = tf.keras.Input(shape=(seq_length,))

            
            # Define the embedding layers
            embedding_layer1 = Embedding(self.vocab_size1, embedding_dim)
            embedding_layer2 = Embedding(self.vocab_size2, embedding_dim)
            embedding_layer3 = Embedding(self.vocab_size3, embedding_dim)
            embedding_layer4 = Embedding(self.vocab_size4, embedding_dim)
            embedding_layer5 = Embedding(self.vocab_size5, embedding_dim)
            embedding_layer6 = Embedding(self.vocab_size6, embedding_dim)
            

            gru_layer = GRU(hidden_units, return_sequences=False)
            
            # Define the dropout layer
            dropout_layer = Dropout(0.3)

            # Define the output layer
            output_layer = Dense(self.vocab_size1, activation='softmax')

            # Connect the layers
            embedded_input1 = embedding_layer1(input_layer1)
            embedded_input2 = embedding_layer2(input_layer2)
            embedded_input3 = embedding_layer3(input_layer3)
            embedded_input4 = embedding_layer4(input_layer4)
            embedded_input5 = embedding_layer5(input_layer5)
            embedded_input6 = embedding_layer6(input_layer6)
            


            input7_cont = Reshape(target_shape=(seq_length, 1))(input_layer7)
            input8_cont = Reshape(target_shape=(seq_length, 1))(input_layer8)

            emb_cont1 = Dense(embedding_dim)(input7_cont)
            emb_cont2 = Dense(embedding_dim)(input8_cont)

            # Concatenate the outputs of the two GRU layers
            concatenated_output = Concatenate()([embedded_input1, embedded_input2, embedded_input3, embedded_input4, embedded_input5, embedded_input6, emb_cont1, emb_cont2])

            gru_output = gru_layer(concatenated_output)


            output = output_layer(gru_output)


            # Create the model
            self.model = Model(inputs=[input_layer1, input_layer2, input_layer3, input_layer4, input_layer5, input_layer6, input_layer7, input_layer8], outputs=output)

            # Compile the model
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])

            # Print the model summary
            self.model.summary()


            
        def _mean_reciprocal_rank(self, recommendations, ground_truth):
            """
            Calculate the Mean Reciprocal Rank (MRR) of a recommendation system.

            :param recommendations: A list of lists containing the recommended items for each query.
            :param ground_truth: A list containing the ground truth (relevant) items for each query.
            :return: The Mean Reciprocal Rank (MRR) value as a float.
            """
            assert len(recommendations) == len(ground_truth), "Recommendations and ground truth lists must have the same length."

            reciprocal_ranks = []

            for rec, gt in zip(recommendations, ground_truth):
                for rank, item in enumerate(rec, start=1):
                    if item == gt:
                        reciprocal_ranks.append(1 / rank)
                        break
                else:
                    reciprocal_ranks.append(0)

            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
            return mrr


        def train(self, epoch= 10, batch_size = 32):
            
            checkpoint_callback = ModelCheckpoint(
                '/kaggle/working/model_checkpoint.h5',
                monitor='val_cosine_similarity',  
                save_best_only=True,  
                save_weights_only=False, 
                verbose=1 
            )
            
            self.history = self.model.fit([ self.X1_train, self.X2_train, self.X3_train, self.X4_train, self.X5_train, self.X6_train, self.X7_train, self.X8_train], self.y_train, epochs = epoch, batch_size=batch_size, verbose = True, validation_split=0.1, callbacks=[checkpoint_callback])
            
        
        
        
        def _decoder(self, recommendation):
            
            '''decode sequeces to ASIN ids'''
            
            decoded = []
            for next_item in recommendation:

                decoded.append([self.items.iloc[e-1] for e in next_item])

            decoded = np.array(decoded)
            return decoded
            
            
            
        def _predictor(self, X_test):
            
            '''generate y_pred (which is top 100 product indices) from the model for X_test. '''
            
            batch_size = 64
            num_batches = int(len(X_test[0]) / batch_size)

            y_pred = []
            for batch_idx in range(num_batches+1):
                
                if batch_idx < num_batches:
                    start_idx = batch_idx * batch_size
                    end_idx = (batch_idx + 1) * batch_size
                    
                    inputs = []
                
                    for i in range(len(X_test)):
                        inputs.append(X_test[i][start_idx:end_idx])
                        
                    predictions = self.model.predict(inputs)
                    recom_size = 100
            
            
                    top_preds = np.argpartition(predictions, -recom_size, axis=1)[:, -recom_size:]
                    sorted_indices = np.argsort(predictions[np.arange(len(predictions))[:, None], top_preds], axis=1)[:, ::-1]
                    recom = top_preds[np.arange(len(predictions))[:, None], sorted_indices]

                    y_pred.append(recom)

                        
                else:
                    
                    inputs = []
                
                    for i in range(len(X_test)):
                        
                        inputs.append(X_test[i][end_idx:])
                    
                    predictions = self.model.predict(inputs)
            
                    top_preds = np.argpartition(predictions, -recom_size, axis=1)[:, -recom_size:]
                    sorted_indices = np.argsort(predictions[np.arange(len(predictions))[:, None], top_preds], axis=1)[:, ::-1]
                    recom = top_preds[np.arange(len(predictions))[:, None], sorted_indices]

                    y_pred.append(recom)
                
            y_pred = [inner_list for outer_list in y_pred for inner_list in outer_list]
                    
            return y_pred
            
            
        def test_1_testontest(self):
            
            
            '''evaluate model's performance on the test set defined in the initialization '''
            #update it for all the test sessions instead of only 200
            
            recommendation = self._predictor([self.X1_test, self.X2_test, self.X3_test, self.X4_test, self.X5_test, self.X6_test, self.X7_test, self.X8_test])
            gnd = self.y_test.tolist()
            self.test1_MRR = self._mean_reciprocal_rank(recommendation, gnd)
            print(f'MRR for test1: {self.test1_MRR}')

    
        
            
        def test_2_testwithendone(self, n=100):
            
            ''' evaluate model's performance on the given test set. Since test set has no ground truth
            we will split the last item in the session and consider it as the next item and evaulate model
            performance'''
            
            
            X1 = self.test_gdf['prev_items-list'].tolist()
            X2 = self.test_gdf['title-list'].tolist()
            X3 = self.test_gdf['brand-list'].tolist()
            X4 = self.test_gdf['size-list'].tolist()
            X5 = self.test_gdf['model-list'].tolist()
            X6 = self.test_gdf['color-list'].tolist()
            
            #handles variable length session sequences
            X1 = np.array(X1, dtype='object')


            #extract next item from the X1: prev_items_list, also remove last items attributes
            X1_p = []
            X2_p = []
            X3_p = []
            X4_p = []
            X5_p = []
            X6_p = []

            y_p = []

            for i in range(len(X1)):
                X1_p.append(X1[i][:-1])
                X2_p.append(X2[i][:-1])
                X3_p.append(X3[i][:-1])
                X4_p.append(X4[i][:-1])
                X5_p.append(X5[i][:-1])
                X6_p.append(X6[i][:-1])
                y_p.append(X1[i][-1])
                
        
            X1 = X1_p
            X2 = X2_p
            X3 = X3_p
            X4 = X4_p
            X5 = X5_p
            X6 = X6_p
            y= y_p
            y = np.array(y)
            
            #padding: pre for X1 and post for all others
            X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
            X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')
            X3 = pad_sequences(X3, maxlen=self.max_len, padding='pre')
            X4 = pad_sequences(X4, maxlen=self.max_len, padding='pre')
            X5 = pad_sequences(X5, maxlen=self.max_len, padding='pre')
            X6 = pad_sequences(X6, maxlen=self.max_len, padding='pre')
            
           
            rec = self._predictor([X1[:n], X2[:n], X3[:n], X4[:n], X5[:n], X6[:n]])
            gnd = y[:n].tolist()
            self.test2_MRR =  self._mean_reciprocal_rank(rec, gnd)
            print(f'MRR for test1: {self.test2_MRR}')

            
        
    
        def test_3_generatefinalresult(self, n=100):
            
            
            ''' generates predictions of test set. Decodes the index and return the recommendations with ASIN ids'''
            
            X1 = self.test_gdf['prev_items-list'].tolist()
            X2 = self.test_gdf['title-list'].tolist()
            X3 = self.test_gdf['brand-list'].tolist()
            X4 = self.test_gdf['size-list'].tolist()
            X5 = self.test_gdf['model-list'].tolist()
            X6 = self.test_gdf['color-list'].tolist()
            X7 = self.test_gdf['price_log_norm-list'].tolist()
            X8 = self.test_gdf['relative_price_to_avg_categ_id-list'].tolist()

            
            X1 = pad_sequences(X1, maxlen=self.max_len, padding='pre')
            X2 = pad_sequences(X2, maxlen=self.max_len, padding='pre')
            X3 = pad_sequences(X3, maxlen=self.max_len, padding='pre')
            X4 = pad_sequences(X4, maxlen=self.max_len, padding='pre')
            X5 = pad_sequences(X5, maxlen=self.max_len, padding='pre')
            X6 = pad_sequences(X6, maxlen=self.max_len, padding='pre')
            X7 = pad_sequences(X7, maxlen=self.max_len, padding='pre')
            X8 = pad_sequences(X8, maxlen=self.max_len, padding='pre')
            
          
            rec = self._predictor([X1[:n], X2[:n], X3[:n], X4[:n], X5[:n], X6[:n],  X7[:n], X8[:n]])

            y_pred = self._decoder(rec)
            y_pred = y_pred.tolist()
            df = pd.DataFrame()
            df['next_item_prediction'] = y_pred
            
            return df
