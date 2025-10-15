import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

// Mock data for demonstration
const data = [
  { time: '10/08', risk: 0.15, threshold: 0.5 },
  { time: '10/09', risk: 0.22, threshold: 0.5 },
  { time: '10/10', risk: 0.35, threshold: 0.5 },
  { time: '10/11', risk: 0.48, threshold: 0.5 },
  { time: '10/12', risk: 0.61, threshold: 0.5 },
  { time: '10/13', risk: 0.68, threshold: 0.5 },
  { time: '10/14', risk: 0.72, threshold: 0.5 },
]

function TrendChart() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          label={{ value: 'Risk Score', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip
          formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
          labelStyle={{ color: '#000' }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="risk"
          stroke="#0066CC"
          strokeWidth={2}
          name="Infection Risk"
          dot={{ fill: '#0066CC', r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="threshold"
          stroke="#DD2C00"
          strokeWidth={2}
          strokeDasharray="5 5"
          name="High Risk Threshold"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default TrendChart

