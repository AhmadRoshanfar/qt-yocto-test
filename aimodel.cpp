#include "aimodel.h"

#define TFLITE_MINIMAL_CHECK(x)                                   \
if (!(x))                                                         \
    {                                                             \
            fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);  \
            exit(1);                                                  \
    }

AIModel::AIModel(QObject *parent)
    : QObject{parent}
{
    // QString projectPath = QDir::currentPath() + "/../../";
    // qDebug()<<projectPath ;
    QString modelPath = QDir::homePath() + "/AiModel/mobilenet.tflite";
    loadLabels(":/AiModel/labels.txt");
    loadModel(modelPath);

    m_interpreter->SetAllowFp16PrecisionForFp32(true);
    m_interpreter->SetNumThreads(1);
}

void AIModel::loadImage()
{
    int input = m_interpreter->inputs()[0];
    auto height = m_interpreter->tensor(input)->dims->data[1];
    auto width = m_interpreter->tensor(input)->dims->data[2];
    auto channels = m_interpreter->tensor(input)->dims->data[3];
            qDebug() << "1";

    // Load Input Image
    cv::Mat image;
    QString imagePath = QDir::homePath() + "/AiModel/owl.jpeg";
    frame = cv::imread(imagePath.toStdString());
    if (frame.empty())
    {
        qDebug() << "Failed to load image";
    }
    // Copy image to input tensor
            qDebug() << "2";

    cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
                qDebug() << "3";

    memcpy(m_interpreter->typed_input_tensor<unsigned char>(0), image.data, image.total() * image.elemSize());
                qDebug() << "4";

}

void AIModel::loadLabels(QString filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Error: Cannot open file";
    }
    // Read the file line by line
    QTextStream in(&file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        m_labels.push_back(line);  // Add the line to the vector
    }
    file.close();
}

void AIModel::loadModel(QString path)
{
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(path.toUtf8().constData());
    TFLITE_MINIMAL_CHECK(model != nullptr);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&m_interpreter);
    TFLITE_MINIMAL_CHECK(m_interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(m_interpreter->AllocateTensors() == kTfLiteOk);
    qDebug() << "=== Pre-invoke Interpreter State ===\n";
    // tflite::PrintInterpreterState(m_interpreter.get());
}

void AIModel::predict()
{
        if (!m_interpreter) {
        qDebug() << "NULL INTERPRETER";
    }
    TFLITE_MINIMAL_CHECK(m_interpreter->Invoke() == kTfLiteOk);
    qDebug() << "\n\n=== Post-invoke Interpreter State ===\n";

    // Get Output
    int output = m_interpreter->outputs()[0];
    TfLiteIntArray *output_dims = m_interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::vector<std::pair<float, int>> top_results;
    float threshold = 0.01f;

    switch (m_interpreter->tensor(output)->type)
    {
    case kTfLiteInt32:
        tflite::label_image::get_top_n<float>(m_interpreter->typed_output_tensor<float>(0), output_size, 1, threshold, &top_results, kTfLiteFloat32);
        break;
    case kTfLiteUInt8:
        tflite::label_image::get_top_n<uint8_t>(m_interpreter->typed_output_tensor<uint8_t>(0), output_size, 1, threshold, &top_results, kTfLiteUInt8);
        break;
    default:
        fprintf(stderr, "cannot handle output type\n");
        exit(-1);
    }
    // Print inference ms in input image
    cv::putText(frame, "Infernce Time in ms: "  /*std::to_string(inference_time)*/, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

    setLabel(m_labels[top_results[0].second] );
    setConfidence( QString::number(top_results[0].first));
}

QString AIModel::label() const
{
    return m_label;
}

void AIModel::setLabel(const QString &newLabel)
{
    if (m_label == newLabel)
    {
        return;
    }
    m_label = newLabel;
    emit labelChanged();
}

QString AIModel::confidence() const
{
    return m_confidence;
}

void AIModel::setConfidence(const QString &newConfidence)
{
    if (m_confidence == newConfidence)
    {
        return;
    }
    m_confidence = newConfidence;
    emit confidenceChanged();
}
