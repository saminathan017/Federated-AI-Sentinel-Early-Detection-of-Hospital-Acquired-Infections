import { Link } from 'react-router-dom'

interface Patient {
  patient_id: string
  encounter_id: string
  infection_risk_score: number
  risk_level: string
  ward: string
}

interface RiskCardProps {
  patient: Patient
}

function RiskCard({ patient }: RiskCardProps) {
  return (
    <div className="card hover:shadow-lg transition">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                Patient {patient.patient_id}
              </h3>
              <p className="text-sm text-gray-500">
                Encounter: {patient.encounter_id} | Ward: {patient.ward}
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Risk Score */}
          <div className="text-right">
            <p className="text-2xl font-bold text-gray-900">
              {(patient.infection_risk_score * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-gray-500">Risk Score</p>
          </div>

          {/* Risk Level Badge */}
          <span
            className={`risk-indicator ${
              patient.risk_level === 'HIGH'
                ? 'risk-high'
                : patient.risk_level === 'MODERATE'
                ? 'risk-moderate'
                : 'risk-low'
            }`}
          >
            {patient.risk_level}
          </span>

          {/* View Details Link */}
          <Link
            to={`/patient/${patient.patient_id}`}
            className="btn-primary text-sm"
          >
            View Details
          </Link>
        </div>
      </div>

      {/* Risk Indicator Bar */}
      <div className="mt-4">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${
              patient.risk_level === 'HIGH'
                ? 'bg-red-600'
                : patient.risk_level === 'MODERATE'
                ? 'bg-yellow-500'
                : 'bg-green-500'
            }`}
            style={{ width: `${patient.infection_risk_score * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export default RiskCard

