<script setup lang="ts">
const props = defineProps<{
  value: string | number
  label: string
}>()
</script>

<template>
  <div class="metric-badge-component">
    <div class="badge-icon">
      <slot name="icon"></slot>
    </div>
    <div class="badge-value">{{ value }}</div>
    <div class="badge-label">{{ label }}</div>
  </div>
</template>

<style scoped>
.metric-badge-component {
  text-align: center;
  padding: 16px;
}

.badge-icon {
  font-size: 1.5rem;
  margin-bottom: 6px;
  line-height: 1;
  color: var(--vp-c-brand-1);
}

.badge-value {
  font-size: 2.5rem;
  font-weight: 800;
  color: var(--vp-c-text-1);
  line-height: 1.1;
  font-family: var(--vp-font-family-mono);
}

.badge-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin-top: 6px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
</style>
