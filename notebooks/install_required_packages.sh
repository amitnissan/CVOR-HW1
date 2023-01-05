# Installing required packages
if pip show yolov7-package | grep Version | grep -q '0.0.12'
then
  echo "yolov7-package already installed"
else
  echo "Installing yolov7-package"
  pip install yolov7-package
  echo "finished installing"
fi