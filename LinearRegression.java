import java.util.List;

public class LinearRegression {
    List<List<Object>> data;
    double[] weights;
    double bias;
    int numFeatures;
    int numSamples;
    LinearRegression() {
        this.data = null;
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] = 0.0;
        }
        this.bias = 0.0;
    }
    private boolean isNumeric(Class<?> cls) {
        return Number.class.isAssignableFrom(cls);
    }
    public void fit(DataFrame dataframe, int targetIndex){
        this.data = dataframe.getData();
        this.numSamples = data.size();
        this.numFeatures = data.get(0).size() - 1;
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        List<Object> target = dataframe.getColumn(targetIndex);
        dataframe.removeColumn(targetIndex);
        checkEncoding(this.data, dataframe.colType);
        fit(this.data, target);
    }
    private void fit(List<List<Object>> features, List<Object> target){
    }
    private void checkEncoding(List<List<Object>> data, List<Class<?>> colType){
        for(int j = 0; j < colType.size(); j++){
            if(!isNumeric(colType.get(j))){
                throw new IllegalArgumentException("All features must be numeric. Column " + j + " is not numeric.");
            }
        }
    }
}
