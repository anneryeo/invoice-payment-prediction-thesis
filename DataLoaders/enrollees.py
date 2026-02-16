class Enrollees():
    def __init__(self, df_revenues):
        self.df_r = df_revenues
        self.df_e = self.load_data()
        
    def load_data(self):
        df_e = self.df_r[['entry_date', 'school_year', 'student_id_pseudonimized', 'category_name', 'amount_due']]
        
        df_e = (
            df_e.groupby(['school_year', 'student_id_pseudonimized', 'category_name'])
            .sum(numeric_only=True)
            .reset_index()
        )
    
        df_e = df_e[(df_e['category_name'].str.contains('-UE')) & (df_e['amount_due'] != 0)]
        df_e['grade_level'] = df_e['category_name'].str[:3]
    
        # Get the minimum date for each ID No. and S.Y.
        enrollment_dates = (
            self.df_r[
                (self.df_r['category_name'].str.contains('-UE')) & (self.df_r['amount_due'] != 0)]
            .groupby(['school_year', 'student_id_pseudonimized'])['entry_date'].min().reset_index()
        )
        
        # Merge back to the main dataframe
        df_e = df_e.merge(enrollment_dates, on=['school_year', 'student_id_pseudonimized'], how='left')
    
        df_e.drop(columns=['amount_due'], inplace=True)
        df_e.rename(columns={'category_name': 'plan_type', 'entry_date': 'enrollment_date'}, inplace=True)
    
        # Define a mapping dictionary for the plan types
        plan_type_mapping = {
            '-A-': 'Plan - A',
            '-B-': 'Plan - B',
            '-C-': 'Plan - C',
            '-D-': 'Plan - D',
            '-E-': 'Plan - E'
        }

        # Function to map plan type
        def map_plan_type(plan_type):
            for pattern, replacement in plan_type_mapping.items():
                if pattern in plan_type:
                    return replacement
            return plan_type

        # Apply the mapping function to the 'plan_type' column
        df_e['plan_type'] = df_e['plan_type'].apply(map_plan_type)

        grade_to_level_mapping= {
            'Nrs': 'Pre-Elementary',
            'Kn1': 'Pre-Elementary',
            'Kn2': 'Pre-Elementary',
            'G01': 'Elementary',
            'G02': 'Elementary',
            'G03': 'Elementary',
            'G04': 'Elementary',
            'G05': 'Elementary',
            'G06': 'Elementary',
            'G07': 'Junior High',
            'G08': 'Junior High',
            'G09': 'Junior High',
            'G10': 'Junior High',
            'SpE': 'Special Education'
        }
        df_e['education_level'] = df_e['grade_level'].map(grade_to_level_mapping)

        
        # Extract refunded enrollees
        df_refunded_enrollees = self.df_r[(self.df_r['category_name'] == "Refund") &
                                          (self.df_r['discount_refund_applied_to'].str.contains('-UE'))]
        df_refunded_enrollees = df_refunded_enrollees[['school_year', 'student_id_pseudonimized']]
        df_refunded_enrollees = \
            df_refunded_enrollees \
            .groupby(['school_year', 'student_id_pseudonimized']) \
            .size() \
            .reset_index(name='Count')
        df_refunded_enrollees.drop(columns=['Count'], inplace=True)
        
        # Create a new column if has refund
        df_e['has_refunded'] = df_e.apply(lambda row: 'Has Refund' \
            if ((df_refunded_enrollees['school_year'] == row['school_year'])
                & (df_refunded_enrollees['student_id_pseudonimized'] == row['student_id_pseudonimized'])).any()
            else 'No Refund', axis=1)
        
        df_e = df_e.reset_index(drop=True)
        return df_e

    def show_data(self):
        return self.df_e