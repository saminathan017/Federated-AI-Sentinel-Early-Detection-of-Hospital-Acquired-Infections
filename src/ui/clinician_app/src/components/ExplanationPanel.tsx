import { useState } from 'react'

interface FeatureContribution {
  feature_name: string
  feature_value: number
  contribution: number
  impact: string
}

interface ExplanationPanelProps {
  patientId: string
}

function ExplanationPanel({ patientId }: ExplanationPanelProps) {
  const [activeTab, setActiveTab] = useState<'drivers' | 'counterfactual'>('drivers')

  // Mock data - in production, fetch from API
  const topDrivers: FeatureContribution[] = [
    {
      feature_name: 'Temperature Mean',
      feature_value: 38.7,
      contribution: 0.25,
      impact: 'increases risk',
    },
    {
      feature_name: 'WBC Count Latest',
      feature_value: 14.5,
      contribution: 0.18,
      impact: 'increases risk',
    },
    {
      feature_name: 'C Reactive Protein',
      feature_value: 85.3,
      contribution: 0.15,
      impact: 'increases risk',
    },
    {
      feature_name: 'Heart Rate Max',
      feature_value: 110.0,
      contribution: 0.12,
      impact: 'increases risk',
    },
    {
      feature_name: 'Lactate Latest',
      feature_value: 2.1,
      contribution: 0.08,
      impact: 'increases risk',
    },
  ]

  const counterfactual = {
    current_risk: 0.72,
    target_risk: 0.30,
    suggested_changes: [
      { feature_name: 'Temperature Mean', current: 38.7, target: 37.5, change: -1.2 },
      { feature_name: 'WBC Count', current: 14.5, target: 10.0, change: -4.5 },
      { feature_name: 'CRP', current: 85.3, target: 40.0, change: -45.3 },
    ],
  }

  return (
    <div className="card">
      <div className="border-b border-gray-200 mb-4">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('drivers')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'drivers'
                ? 'border-sentinel-blue text-sentinel-blue'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Top Clinical Drivers
          </button>
          <button
            onClick={() => setActiveTab('counterfactual')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'counterfactual'
                ? 'border-sentinel-blue text-sentinel-blue'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            What-If Scenarios
          </button>
        </nav>
      </div>

      {activeTab === 'drivers' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Features contributing most to infection risk prediction for Patient {patientId}
          </p>

          {topDrivers.map((driver, index) => (
            <div key={index} className="border-l-4 border-sentinel-blue pl-4 py-2">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900">{driver.feature_name}</h4>
                  <p className="text-sm text-gray-600">
                    Value: {driver.feature_value.toFixed(2)} → {driver.impact}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold text-gray-900">
                    {(driver.contribution * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-500">contribution</p>
                </div>
              </div>
              <div className="mt-2 w-full bg-gray-200 rounded-full h-1.5">
                <div
                  className="bg-sentinel-blue h-1.5 rounded-full"
                  style={{ width: `${driver.contribution * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'counterfactual' && (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm font-medium text-blue-900">
              To reduce risk from {(counterfactual.current_risk * 100).toFixed(1)}% to{' '}
              {(counterfactual.target_risk * 100).toFixed(1)}%, consider these interventions:
            </p>
          </div>

          {counterfactual.suggested_changes.map((change, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <h4 className="font-medium text-gray-900">{change.feature_name}</h4>
                <p className="text-sm text-gray-600">
                  {change.current.toFixed(2)} → {change.target.toFixed(2)}
                </p>
              </div>
              <div className="text-right">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                  {change.change > 0 ? '+' : ''}
                  {change.change.toFixed(2)}
                </span>
              </div>
            </div>
          ))}

          <div className="mt-4 text-sm text-gray-500 italic">
            These are model-generated suggestions. Always validate with clinical assessment.
          </div>
        </div>
      )}
    </div>
  )
}

export default ExplanationPanel

