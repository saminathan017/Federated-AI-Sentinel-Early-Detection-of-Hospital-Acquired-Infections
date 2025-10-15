import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import PatientView from './pages/PatientView'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Federated AI Sentinel</h1>
                <p className="text-sm text-gray-500">Early Infection Detection System</p>
              </div>
              <nav className="flex space-x-4">
                <Link
                  to="/"
                  className="text-gray-700 hover:text-sentinel-blue px-3 py-2 rounded-md text-sm font-medium"
                >
                  Dashboard
                </Link>
                <Link
                  to="/patient/demo"
                  className="text-gray-700 hover:text-sentinel-blue px-3 py-2 rounded-md text-sm font-medium"
                >
                  Patient View
                </Link>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/patient/:patientId" element={<PatientView />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <p className="text-center text-sm text-gray-500">
              Research prototype. Not for clinical decision-making without validation.
            </p>
          </div>
        </footer>
      </div>
    </Router>
  )
}

export default App

