import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors, FontSize, BorderRadius } from '../theme';

interface StepIndicatorProps {
  steps: string[];
  currentStep: number;
}

const StepIndicator: React.FC<StepIndicatorProps> = ({ steps, currentStep }) => {
  return (
    <View style={styles.container}>
      {steps.map((step, index) => (
        <React.Fragment key={index}>
          <View style={styles.stepWrapper}>
            <View style={[
              styles.circle,
              index < currentStep && styles.completedCircle,
              index === currentStep && styles.activeCircle,
            ]}>
              {index < currentStep ? (
                <Text style={styles.checkmark}>&#10003;</Text>
              ) : (
                <Text style={[
                  styles.stepNumber,
                  index === currentStep && styles.activeStepNumber,
                ]}>{index + 1}</Text>
              )}
            </View>
            <Text style={[
              styles.stepLabel,
              index === currentStep && styles.activeLabel,
              index < currentStep && styles.completedLabel,
            ]} numberOfLines={1}>{step}</Text>
          </View>
          {index < steps.length - 1 && (
            <View style={[styles.line, index < currentStep && styles.completedLine]} />
          )}
        </React.Fragment>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'center',
    paddingHorizontal: 16,
  },
  stepWrapper: {
    alignItems: 'center',
    width: 60,
  },
  circle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  activeCircle: {
    backgroundColor: Colors.primary,
  },
  completedCircle: {
    backgroundColor: Colors.success,
  },
  stepNumber: {
    fontSize: FontSize.sm,
    fontWeight: '700',
    color: Colors.textSecondary,
  },
  activeStepNumber: {
    color: Colors.white,
  },
  checkmark: {
    fontSize: FontSize.sm,
    fontWeight: '700',
    color: Colors.white,
  },
  stepLabel: {
    fontSize: 10,
    color: Colors.textSecondary,
    textAlign: 'center',
  },
  activeLabel: {
    color: Colors.primary,
    fontWeight: '700',
  },
  completedLabel: {
    color: Colors.success,
    fontWeight: '600',
  },
  line: {
    flex: 1,
    height: 2,
    backgroundColor: '#E5E7EB',
    marginTop: 15,
    marginHorizontal: -4,
  },
  completedLine: {
    backgroundColor: Colors.success,
  },
});

export default StepIndicator;
