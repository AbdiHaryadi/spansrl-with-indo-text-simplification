@echo off

set "MODEL_PATH=.\models"
if not exist "%MODEL_PATH%" (
  mkdir "%MODEL_PATH%"
)

set "RESULTS_PATH=.\data\results"
if not exist "%RESULTS_PATH%" (
  mkdir "%RESULTS_PATH%"
)
