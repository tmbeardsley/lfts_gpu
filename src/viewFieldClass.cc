
#include "viewFieldClass.h"



// Constructor
viewFieldClass::viewFieldClass(int *m, int vidSampleRate) :
    bool_update(false),
    M(m[0]*m[1]*m[2])
{

    // Create an array to hold the field
    field_cpu_ = std::make_unique<double[]>(M);

    // Wrap the raw array in a vtkDoubleArray
    this->scalarData = vtkSmartPointer<vtkDoubleArray>::New();
    this->scalarData->SetArray(field_cpu_.get(), M, 1);    // https://vtk.org/doc/nightly/html/classvtkAOSDataArrayTemplate.html#a16c9ad0bf4b6b469a3c1fb048d98b3d0

    // Create vtkStructuredPoints and assign scalar data
    this->structuredPoints = vtkSmartPointer<vtkStructuredPoints>::New();
    this->structuredPoints->SetDimensions(m[0], m[1], m[2]);
    this->structuredPoints->GetPointData()->SetScalars(this->scalarData);

    // Create transfer mapping scalar value to opacity
    this->opacityTransferFunction = vtkSmartPointer<vtkPiecewiseFunction>::New();
    this->opacityTransferFunction->AddPoint(-0.97, 0.0);
    this->opacityTransferFunction->AddPoint(0.96, 1.0);

    // Create transfer mapping scalar value to color
    this->colorTransferFunction = vtkSmartPointer<vtkColorTransferFunction>::New();
    this->colorTransferFunction->AddRGBPoint(-1, 0.231373, 0.289039, 0.752941); // Blue
    this->colorTransferFunction->AddRGBPoint(0, 0.865003, 0.865003, 0.865003);  // White
    this->colorTransferFunction->AddRGBPoint(1, 0.705882, 0.0156863, 0.14902);  // Red

    // The properties of the volume describe how the data will look.
    this->volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
    this->volumeProperty->SetColor(this->colorTransferFunction);
    this->volumeProperty->SetScalarOpacity(this->opacityTransferFunction);
    this->volumeProperty->SetInterpolationTypeToLinear();

    // The mapper / ray cast function knows how to render the data
    this->volumeMapper = vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper>::New();
    this->volumeMapper->SetInputData(this->structuredPoints);

    // The volume holds the volumemapper and volumeproperty and
    // can be used to position/orient the volume
    this->volume = vtkSmartPointer<vtkVolume>::New();
    this->volume->SetMapper(this->volumeMapper);
    this->volume->SetProperty(this->volumeProperty);

    // Create a NamedColors instance
    vtkNew<vtkNamedColors> colors;

    // Create a new renderer instance
    this->ren = vtkSmartPointer<vtkRenderer>::New();
    this->ren->AddVolume(this->volume);
    this->ren->SetBackground(colors->GetColor3d("White").GetData());
    this->ren->GetActiveCamera()->Azimuth(45);
    this->ren->GetActiveCamera()->Elevation(30);
    this->ren->ResetCameraClippingRange();
    this->ren->ResetCamera();

    // Add the renderer to a new render window instance
    this->renWin = vtkSmartPointer<vtkRenderWindow>::New();
    this->renWin->AddRenderer(this->ren);
    // Enable double buffering
    this->renWin->DoubleBufferOn();

    // Associate the render window with the interactor
    this->interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    this->interactor->SetRenderWindow(this->renWin);

    // Set the interactor style and adjust the rotation factor sensitivity
    style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    style->SetMotionFactor(5.0);
    this->interactor->SetInteractorStyle(style);

    // Set the size and title of the render window
    this->renWin->SetSize(600, 600);
    this->renWin->SetWindowName("View Configuration");

    interactor->Initialize();

    // Create a new UpdateCallback class instance and initialise it with a pointer
    // to the current viewFieldClass instance (this), and the vtkDoubleArray
    // instance that wraps the field_cpu_[] array (scalarData).
    callback = vtkSmartPointer<UpdateCallback>::New();
    callback->Initialize(this, scalarData);

    // Add the callback to the interactor and call it every 50ms to 
    // re-render the scene if necessary.
    interactor->AddObserver(vtkCommand::TimerEvent, callback);
    interactor->CreateRepeatingTimer(100);

    // Add an observer for the window close event to stop the interactor
    interactor->AddObserver(vtkCommand::ExitEvent, this, &viewFieldClass::OnWindowClose);

}

// Public function to start the interactor
void viewFieldClass::startInteractor() {
    interactor->Start();
}

// Method to handle window close event
void viewFieldClass::OnWindowClose() {
    interactor->TerminateApp();
    interactor->GetRenderWindow()->Finalize();
}

// Destructor
viewFieldClass::~viewFieldClass() {
}

// Getter for bool_update
bool viewFieldClass::getBoolUpdate() const {
    return bool_update;
}

// Setter for bool_update
void viewFieldClass::setBoolUpdate(bool value) {
    bool_update = value;
}

// Return a reference to the class-level mutex variable for locking
std::mutex& viewFieldClass::getMutex() {
    return field_cpu_mutex_;
}

// Pointer to the field array memory so it can be updated directly from the gpu
// from outside of the class
double* viewFieldClass::field_arr_ptr() {
    return field_cpu_.get();
}






// Static method to return a pointer to a new instance of UpdateCallback
UpdateCallback* UpdateCallback::New() {
    return new UpdateCallback();
}

// Keep a pointer to an instance of a viewFieldClass to access the rendering setup and data to visualise
void UpdateCallback::Initialize(viewFieldClass* instance, vtkDoubleArray* ScalarData) {
    this->viewFieldClassInstance = instance;
    this->ScalarData = ScalarData;
}

// Code to be executed each time the TimerEvent (to which the UpdateCallback instance is bound) fires
void UpdateCallback::Execute(vtkObject* caller, unsigned long eventId, void* callData) {

    if (eventId == vtkCommand::TimerEvent) {

        // Check if an update is needed
        if (viewFieldClassInstance->getBoolUpdate()) {

            {
                // ScalarData is a wrapper for field_cpu_, so get lock on the mutex to
                // prevent reading/writing race conditions.
                std::lock_guard<std::mutex> lock(viewFieldClassInstance->getMutex());

                // Notify VTK that the data has changed
                ScalarData->Modified();

                // Re-render the scene
                vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
                interactor->GetRenderWindow()->Render();
            }

            // Reset the update flag to show that the data in viewFieldClass::field_cpu_[] has been plotted
            viewFieldClassInstance->setBoolUpdate(false);
        }
    }
}



