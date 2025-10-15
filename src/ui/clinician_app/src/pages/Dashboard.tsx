import { useState, useEffect } from 'react'
import RiskCard from '../components/RiskCard'
import TrendChart from '../components/TrendChart'

interface Patient {
  patient_id: string
  encounter_id: string
  infection_risk_score: number
  risk_level: string
  ward: string
}

function Dashboard() {
  const [patients, setPatients] = useState<Patient[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading patient data
    // In production, fetch from API
    setTimeout(() => {
      setPatients([
        {
          patient_id: 'P001234',
          encounter_id: 'E001',
          infection_risk_score: 0.72,
          risk_level: 'HIGH',
          ward: 'ICU-2',
        },
        {
          patient_id: 'P001235',
          encounter_id: 'E002',
          infection_risk_score: 0.38,
          risk_level: 'MODERATE',
          ward: 'Ward-3A',
        },
        {
          patient_id: 'P001236',
          encounter_id: 'E003',
          infection_risk_score: 0.12,
          risk_level: 'LOW',
          ward: 'Ward-2B',
        },
      ])
      setLoading(false)
    }, 500)
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading dashboard...</div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">High Risk Patients</h3>
          <p className="mt-2 text-3xl font-bold text-red-600">
            {patients.filter((p) => p.risk_level === 'HIGH').length}
          </p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Moderate Risk</h3>
          <p className="mt-2 text-3xl font-bold text-yellow-600">
            {patients.filter((p) => p.risk_level === 'MODERATE').length}
          </p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Monitored</h3>
          <p className="mt-2 text-3xl font-bold text-gray-900">{patients.length}</p>
        </div>
      </div>

      {/* Risk Trend Chart */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Infection Risk Trend (Last 7 Days)
        </h2>
        <TrendChart />
      </div>

      {/* Patient List */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">Active Patients</h2>
        {patients.map((patient) => (
          <RiskCard key={patient.patient_id} patient={patient} />
        ))}
      </div>
    </div>
  )
}

export default Dashboard

