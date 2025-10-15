import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import ExplanationPanel from '../components/ExplanationPanel'
import TrendChart from '../components/TrendChart'

interface PatientDetails {
  patient_id: string
  encounter_id: string
  infection_risk_score: number
  risk_level: string
  ward: string
  age: number
  admission_date: string
}

function PatientView() {
  const { patientId } = useParams()
  const [patient, setPatient] = useState<PatientDetails | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading patient details
    setTimeout(() => {
      setPatient({
        patient_id: patientId || 'demo',
        encounter_id: 'E001',
        infection_risk_score: 0.72,
        risk_level: 'HIGH',
        ward: 'ICU-2',
        age: 68,
        admission_date: '2025-10-10',
      })
      setLoading(false)
    }, 500)
  }, [patientId])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading patient data...</div>
      </div>
    )
  }

  if (!patient) {
    return (
      <div className="card">
        <p className="text-gray-500">Patient not found</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Patient Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Patient {patient.patient_id}
            </h1>
            <p className="text-gray-500">
              Encounter: {patient.encounter_id} | Ward: {patient.ward}
            </p>
          </div>
          <div>
            <span
              className={`risk-indicator ${
                patient.risk_level === 'HIGH'
                  ? 'risk-high'
                  : patient.risk_level === 'MODERATE'
                  ? 'risk-moderate'
                  : 'risk-low'
              }`}
            >
              {patient.risk_level} RISK
            </span>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-500">Age</p>
            <p className="text-lg font-semibold">{patient.age} years</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Admission</p>
            <p className="text-lg font-semibold">{patient.admission_date}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Risk Score</p>
            <p className="text-lg font-semibold">
              {(patient.infection_risk_score * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Risk Trend */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Risk Trend</h2>
        <TrendChart />
      </div>

      {/* Explanation */}
      <ExplanationPanel patientId={patient.patient_id} />
    </div>
  )
}

export default PatientView

