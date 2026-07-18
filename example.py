from sklearn.datasets import load_boston  
from sklearn.model_selection import train_test_split  


# Load the Boston Housing dataset  
boston = load_boston()   
X = boston.data  
y = boston.target  
# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,   random_state=42)  



# Scale the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  


# Create  the MLP model  
model = Sequential([ Dense(32, activation='relu', input_dim=X_train.shape[1]), Dense(16, 
activation='relu'), Dense(1) ])  


# Compile the model  
model.compile(optimizer='adam', loss='mean_squared_error')  


# Train the model  
model.fit(X_train_scaled,  y_train,  epochs=100, batch_size=32, 
validation_data=(X_test_scaled, y_test))  


# Evaluate the model  
test_loss = model.evaluate(X_test_scaled, y_test)  
print('Test Loss:', test_loss) 


# Make predictions  
y_pred = model.predict(X_test_scaled)  


# Visualize the predictions  
plt.scatter(y_test, y_pred)  
plt.xlabel('Actual Prices')  
plt.ylabel('Predicted Prices')  
plt.title('Actual  vs. Predicted House Prices') 
plt.show()  