import java.util.List;

public class LinearRegression {
    List<List<Object>> data;
    double[] weights;
    double bias;
    int numFeatures;
    int numSamples;
    EncodeData encoder;
    double learningRate = 0.01;
    int epochs = 1000;
    LinearRegression() {
        this.data = null;
        this.bias = 0.0;
    }
    private boolean isNumeric(Class<?> cls) {
        return Number.class.isAssignableFrom(cls);
    }
    private void checkEncoding(List<List<Object>> data, List<Class<?>> colType){
        if (this.encoder == null) {
            this.encoder = new EncodeData();
        }
        int numCols = colType.size();
        for(int j = 0; j < numCols; j++){
            if(isNumeric(colType.get(j))){
                continue;
            }
            String key = "col_" + j;
            this.encoder.newMap(key);

            int code = 0;

            for(int i = 0; i < data.size(); i++){
                Object val = data.get(i).get(j);
                String subKey = val.toString();
                if(!this.encoder.checkSubkey(key, subKey)){
                    this.encoder.addSingleData(key, subKey, code);
                    code += 1;
                }
                int encodedVal = this.encoder.getData(key, subKey);
                data.get(i).set(j, encodedVal);
            }
        }
    }
    public void fit(DataFrame dataframe, int targetIndex){
        this.data = dataframe.getData();
        this.numSamples = data.size();
        this.numFeatures = data.get(0).size() - 1;
        this.weights = new double[numFeatures];
        List<Object> target = dataframe.getColumn(targetIndex);
        dataframe.removeColumn(targetIndex);
        checkEncoding(this.data, dataframe.colType);
        fit(this.data, target);
    }
    private double predictRow(List<Object> row) {
        double yHat =this.bias;
        for (int j = 0; j < this.weights.length; j++) {
            yHat += this.weights[j] * ((Number) row.get(j)).doubleValue();
        }
        return yHat;
    }
    private void fit(List<List<Object>> features, List<Object> target){
        for(int epoch = 0; epoch < epochs; epoch++){
            double[] weightGradients = new double[numFeatures];
            double biasGradient = 0.0;
            for(int i = 0; i < numSamples; i++){
                double yHat = predictRow(features.get(i));
                double yTrue = ((Number) target.get(i)).doubleValue();
                double error = yHat - yTrue;
                for(int j = 0; j < numFeatures; j++){
                    weightGradients[j] += error * ((Number) features.get(i).get(j)).doubleValue();
                }
                biasGradient += error;
            }
            double loss = 0.0;
            for (int i = 0; i < numSamples; i++) {
                double err = predictRow(features.get(i)) -
                            ((Number) target.get(i)).doubleValue();
                loss += err * err;
            }
            loss /= numSamples;
            if (epoch % 10 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + loss);
            }
            for(int j = 0; j < numFeatures; j++){
                this.weights[j] -= (learningRate / numSamples) * weightGradients[j];
            }
           this.bias -= (learningRate / numSamples) * biasGradient;
        }
    }

    public double predict(List<Object> row){
        if (row.size() != numFeatures) {
            throw new IllegalArgumentException("Feature size mismatch");
        }
        List<Object> encodedRow = new java.util.ArrayList<>(numFeatures);
        for(int i = 0; i < numFeatures; i++){
            Object val = row.get(i);
            if(val instanceof Number){
                encodedRow.add(((Number) val).doubleValue());
                continue;
            }
            String key = "col_" + i;
            String subKey = val.toString();
            if(!this.encoder.checkSubkey(key, subKey)){
                throw new IllegalArgumentException("Unknown category: " + subKey + " in column " + i);
            }
            double encodedVal = (double) this.encoder.getData(key, subKey);
            encodedRow.add(encodedVal);
        }
        return predictRow(encodedRow);
    }
    public double[] predict(DataFrame dataframe) {
        List<List<Object>> features = dataframe.getData();

        int n = features.size();
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++) {
            predictions[i] = predict(features.get(i));
        }

        return predictions;
    }
}
