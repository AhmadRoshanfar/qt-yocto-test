#ifndef AIMODEL_H
#define AIMODEL_H

#include <QObject>
#include <QVector>
#include <QString>
#include <QFile>
#include <QDebug>
#include <QUrl>
#include <QDir>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"

class AIModel : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString label READ label WRITE setLabel NOTIFY labelChanged FINAL)
    Q_PROPERTY(QString confidence READ confidence WRITE setConfidence NOTIFY confidenceChanged FINAL)

public:
    explicit AIModel(QObject *parent = nullptr);
    void loadLabels(QString filePath);
    void loadModel(QString path);

    Q_INVOKABLE void loadImage(QUrl path);
    Q_INVOKABLE void predict();

    QString label() const;
    void setLabel(const QString &newLabel);

    QString confidence() const;
    void setConfidence(const QString &newConfidence);

signals:
    void labelChanged();
    void confidenceChanged();

private:
    QVector<QString> m_labels;
    std::unique_ptr<tflite::Interpreter> m_interpreter;
    cv::Mat frame;
    QString m_label;
    QString m_confidence;
};

#endif // AIMODEL_H
