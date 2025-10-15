/**
 * Heatmap component for visualizing risk across wards.
 * 
 * Shows color-coded cells for each ward/time combination.
 */

interface HeatmapProps {
  data?: Array<{ ward: string; time: string; risk: number }>
}

function Heatmap({ data }: HeatmapProps) {
  // Mock data if none provided
  const mockData = data || [
    { ward: 'ICU-1', time: '00:00', risk: 0.35 },
    { ward: 'ICU-1', time: '08:00', risk: 0.42 },
    { ward: 'ICU-1', time: '16:00', risk: 0.58 },
    { ward: 'ICU-2', time: '00:00', risk: 0.65 },
    { ward: 'ICU-2', time: '08:00', risk: 0.72 },
    { ward: 'ICU-2', time: '16:00', risk: 0.68 },
    { ward: 'Ward-3A', time: '00:00', risk: 0.15 },
    { ward: 'Ward-3A', time: '08:00', risk: 0.22 },
    { ward: 'Ward-3A', time: '16:00', risk: 0.18 },
  ]

  const getRiskColor = (risk: number): string => {
    if (risk >= 0.5) return 'bg-red-500'
    if (risk >= 0.2) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  const wards = Array.from(new Set(mockData.map((d) => d.ward)))
  const times = Array.from(new Set(mockData.map((d) => d.time)))

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">
        Risk Heatmap by Ward
      </h2>

      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr>
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-500">Ward</th>
              {times.map((time) => (
                <th key={time} className="px-4 py-2 text-center text-sm font-medium text-gray-500">
                  {time}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {wards.map((ward) => (
              <tr key={ward}>
                <td className="px-4 py-2 text-sm font-medium text-gray-900">{ward}</td>
                {times.map((time) => {
                  const cell = mockData.find((d) => d.ward === ward && d.time === time)
                  const risk = cell?.risk || 0
                  return (
                    <td key={time} className="px-4 py-2">
                      <div
                        className={`w-16 h-10 rounded flex items-center justify-center text-white text-sm font-medium ${getRiskColor(
                          risk
                        )}`}
                      >
                        {(risk * 100).toFixed(0)}%
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex items-center space-x-4 text-sm">
        <span className="flex items-center">
          <div className="w-4 h-4 bg-green-500 rounded mr-2" />
          Low (0-20%)
        </span>
        <span className="flex items-center">
          <div className="w-4 h-4 bg-yellow-500 rounded mr-2" />
          Moderate (20-50%)
        </span>
        <span className="flex items-center">
          <div className="w-4 h-4 bg-red-500 rounded mr-2" />
          High (50%+)
        </span>
      </div>
    </div>
  )
}

export default Heatmap

