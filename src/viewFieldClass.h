// ##################################################################################################
// Defines the viewFieldClass class for organising data visualisation.
// Defines the UpdateCallback class (a subclass of vtkCommand) to periodically refresh the plot.
// The data that is visualised is written to / read from the field_cpu_[] array in viewFieldClass.
// viewFieldClass runs an instance of UpdateCallback on a separate thread, so viewFieldClass 
// contains a mutex (field_cpu_mutex_) that must be locked when writing to / reading from field_cpu_.
// To prevent the callback from attempting to replot consumed data, bool_update in viewFieldClass 
// should be set to true each time new data is sent to the field_cpu_[] array, which is achieved via
// the public member function, 
// ##################################################################################################
#pragma once

#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkOpenGLGPUVolumeRayCastMapper.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkStructuredPoints.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <atomic>
#include <thread>
#include <mutex>
#include <memory>



// Forward declaration of UpdateCallback
class UpdateCallback;

class viewFieldClass {
    public:
        // VTK library classes required for rendering a 3d field
        vtkSmartPointer<vtkDoubleArray> scalarData;
        vtkSmartPointer<vtkStructuredPoints> structuredPoints;
        vtkSmartPointer<vtkRenderer> ren;
        vtkSmartPointer<vtkRenderWindow> renWin;
        vtkSmartPointer<vtkRenderWindowInteractor> interactor;
        vtkSmartPointer<vtkPiecewiseFunction> opacityTransferFunction;
        vtkSmartPointer<vtkColorTransferFunction> colorTransferFunction;
        vtkSmartPointer<vtkVolumeProperty> volumeProperty;
        vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper> volumeMapper;
        vtkSmartPointer<vtkVolume> volume;
        vtkSmartPointer<UpdateCallback> callback;
        vtkSmartPointer<vtkInteractorStyleTrackballCamera> style;

        std::mutex field_cpu_mutex_;                // Lockable mutex to prevent race condition on reading/writing field_cpu_[]
        std::thread interactionThread;              // Reference to a thread that will be running an UpdateCallback
        std::unique_ptr<double[]> field_cpu_;       // The data to be visualised
        std::atomic<bool> bool_update;              // Flag should be set to true to indicate new data is available to visualise
        const int M;
        

    public:
        // Constructor
        viewFieldClass(int *m, int vidSampleRate = -1);

        // Destructor
        ~viewFieldClass();

        // Getter for bool_update
        bool getBoolUpdate() const;

        // Setter for bool_update
        void setBoolUpdate(bool value);

        // Provide a reference to the lockable mutex
        std::mutex& getMutex();

        // Pointer to the field array memory (so it can be updated directly from the gpu)
        double* field_arr_ptr();

        // Public function to start the interactor
        void startInteractor();
    
    private:
        // Method to handle window close event
        void OnWindowClose();
};




// UpdateCallback class definition - subclass of vtkCommand
class UpdateCallback : public vtkCommand {

    public:
        // Return a pointer to a new UpdateCallback instance
        static UpdateCallback* New();

        // Code that will attempt to re-render the scene upon a vtkCommand::TimerEvent
        void Execute(vtkObject* caller, unsigned long eventId, void* callData) override;

        // Initialise the UpdateCallback instance with pointers to a viewFieldClass and
        // associated vtkDoubleArray for rendering.
        void Initialize(viewFieldClass* instance, vtkDoubleArray* ScalarData);

    private:
        viewFieldClass* viewFieldClassInstance;
        vtkDoubleArray* ScalarData;
};


