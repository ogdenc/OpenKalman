/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMATION_HPP
#define OPENKALMAN_LINEARTRANSFORMATION_HPP


namespace OpenKalman
{

  /**
   * \brief A linear transformation from one single-column vector to another.
   *
   * \tparam InputCoefficients Coefficient types for the input.
   * \tparam OutputCoefficients Coefficient types for the output.
   * \tparam TransformationMatrix Transformation matrix. It is a native matrix type with rows corresponding to
   * OutputCoefficients and columns corresponding to InputCoefficients.
   * \tparam PerturbationTransformationMatrices Transformation matrices for each potential perturbation term.
   * if the parameter is not given, the transformation matrix is assumed to be identity (i.e., it is a translation).
   * It is a native matrix type with both rows and columns corresponding to OutputCoefficients.
   */
  template<
    typename InputCoefficients,
    typename OutputCoefficients,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransformation;


  template<typename In, typename Out, typename Tm, typename...Pm, std::size_t order>
  struct is_linearized_function<LinearTransformation<In, Out, Tm, Pm...>, order,
    std::enable_if_t<order <= 2>> : std::true_type {};


  template<
    typename InputCoefficients_,
    typename OutputCoefficients_,
    typename TransformationMatrix,
    typename ... PerturbationTransformationMatrices>
  struct LinearTransformation
  {
    using InputCoefficients = InputCoefficients_;
    using OutputCoefficients = OutputCoefficients_;
    static_assert(typed_matrix_base<TransformationMatrix>);
    static_assert((typed_matrix_base<PerturbationTransformationMatrices> and ...));
    static_assert(MatrixTraits<TransformationMatrix>::dimension == OutputCoefficients::size);
    static_assert(MatrixTraits<TransformationMatrix>::columns == InputCoefficients::size);
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::dimension == OutputCoefficients::size) and ...));
    static_assert(((MatrixTraits<PerturbationTransformationMatrices>::columns == OutputCoefficients::size) and ...));

  protected:
    template<typename T, typename ColumnCoefficients = OutputCoefficients>
    static constexpr bool is_valid_input_matrix_v()
    {
      static_assert(typed_matrix<T> or typed_matrix_base<T>);
      if constexpr(typed_matrix<T>)
        return
          equivalent_to<typename MatrixTraits<T>::RowCoefficients, OutputCoefficients> and
          equivalent_to<typename MatrixTraits<T>::ColumnCoefficients, ColumnCoefficients>;
      else
        return
          MatrixTraits<T>::dimension == OutputCoefficients::size and
          MatrixTraits<T>::columns == ColumnCoefficients::size;
    }

    template<typename Jacobians, typename InputTuple, std::size_t...ints>
    constexpr auto sumprod(Jacobians&& js, InputTuple&& inputs, std::index_sequence<ints...>) const
    {
      return make_self_contained(
        ((std::get<ints>(std::forward<Jacobians>(js)) * std::get<ints>(std::forward<InputTuple>(inputs))) + ...));
    }

    using TransformationMatricesTuple = std::tuple<
      const Matrix<OutputCoefficients, InputCoefficients, self_contained_t<TransformationMatrix>>,
      const Matrix<OutputCoefficients, OutputCoefficients, self_contained_t<PerturbationTransformationMatrices>>...>;
    const TransformationMatricesTuple transformation_matrices;

  public:
    LinearTransformation(const TransformationMatrix& mat, const PerturbationTransformationMatrices& ... p_mats)
      : transformation_matrices(mat, p_mats...) {}

#ifdef __cpp_concepts
    template<typename T, typename ... Ps> requires
      (typed_matrix<T> or typed_matrix_base<T>) and ((typed_matrix<Ps> or typed_matrix_base<Ps>) and ...)
#else
    template<typename T, typename ... Ps, std::enable_if_t<
      (typed_matrix<T> or typed_matrix_base<T>) and
      ((typed_matrix<Ps> or typed_matrix_base<Ps>) and ...), int> = 0>
#endif
    LinearTransformation(T&& mat, Ps&& ... p_mats) noexcept
      : transformation_matrices(std::forward<T>(mat), std::forward<Ps>(p_mats)...)
    {
      static_assert(is_valid_input_matrix_v<T, InputCoefficients>());
      static_assert((is_valid_input_matrix_v<Ps, OutputCoefficients>() and ...));
    }

    /// Applies the transformation.
#ifdef __cpp_concepts
    template<column_vector In, perturbation ... Perturbations> requires
    internal::transformation_args<In, Perturbations...>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      column_vector<In> and (perturbation<Perturbations> and ...) and
      internal::transformation_args<In, Perturbations...>, int> = 0>
#endif
    auto operator()(const In& in, const Perturbations& ... ps) const
    {
      return sumprod(
        jacobian(in, ps...),
        std::forward_as_tuple(in, internal::get_perturbation(ps)...),
        std::make_index_sequence<sizeof...(Perturbations) + 1>{});
    }

    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<column_vector In, perturbation ... Perturbations> requires
    internal::transformation_args<In, Perturbations...>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      column_vector<In> and (perturbation<Perturbations> and ...) and
      internal::transformation_args<In, Perturbations...>, int> = 0>
#endif
    auto jacobian(const In&, const Perturbations&...) const
    {
      constexpr auto mat_count = std::tuple_size_v<TransformationMatricesTuple>;

      // If there are more inputs than transformation matrices, pad the list with extra identity matrices.
      if constexpr(sizeof...(Perturbations) + 1 > mat_count)
      {
        constexpr auto pad_size = sizeof...(Perturbations) + 1 - mat_count;
        auto id = make_Matrix<OutputCoefficients, OutputCoefficients>(MatrixTraits<TransformationMatrix>::identity());
        return std::tuple_cat(transformation_matrices, internal::tuple_replicate<pad_size>(id));
      }
      else
      {
        return internal::tuple_slice<0, sizeof...(Perturbations) + 1>(transformation_matrices);
      }
    }

    /// Returns a tuple of Hessian matrices for the input and each perturbation term. In this case, they are zero matrices.
#ifdef __cpp_concepts
    template<column_vector In, perturbation ... Perturbations> requires
    internal::transformation_args<In, Perturbations...>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      column_vector<In> and (perturbation<Perturbations> and ...) and
      internal::transformation_args<In, Perturbations...>, int> = 0>
#endif
    auto hessian(const In&, const Perturbations&...) const
    {
      return zero_hessian<OutputCoefficients, In, Perturbations...>();
    }

  };


  /*
   * Deduction guides
   */

#ifdef __cpp_concepts
  template<typename T, typename ... Ps> requires
    (typed_matrix<T> and ... and (typed_matrix<Ps> or typed_matrix_base<Ps>))
#else
  template<typename T, typename ... Ps, std::enable_if_t<
    (typed_matrix<T> and ... and (typed_matrix<Ps> or typed_matrix_base<Ps>)), int> = 0>
#endif
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    typename MatrixTraits<T>::ColumnCoefficients,
    typename MatrixTraits<T>::RowCoefficients,
    self_contained_t<typename MatrixTraits<T>::BaseMatrix>,
    std::conditional_t<
      typed_matrix<Ps>,
      self_contained_t<typename MatrixTraits<Ps>::BaseMatrix>,
      self_contained_t<std::decay_t<Ps>>>...>;


#ifdef __cpp_concepts
  template<typed_matrix_base T, typed_matrix_base ... Ps>
#else
  template<typename T, typename ... Ps, std::enable_if_t<
    (typed_matrix_base<T> and ... and typed_matrix_base<Ps>), int> = 0>
#endif
  LinearTransformation(T&&, Ps&& ...)
  -> LinearTransformation<
    Axes<MatrixTraits<T>::columns>,
    Axes<MatrixTraits<T>::dimension>,
    self_contained_t<std::decay_t<T>>,
    self_contained_t<std::decay_t<Ps>>...>;


  /*
   * Traits
   */

  template<typename InC, typename OutC, typename T, typename ... Ps>
  struct is_linearized_function<LinearTransformation<InC, OutC, T, Ps...>, 0> : std::true_type {};

  template<typename InC, typename OutC, typename T, typename ... Ps>
  struct is_linearized_function<LinearTransformation<InC, OutC, T, Ps...>, 1> : std::true_type
  {
    static constexpr auto get_lambda(const LinearTransformation<InC, OutC, T, Ps...>& t)
    {
      return [&t] (auto&&...inputs) { return t.jacobian(std::forward<decltype(inputs)>(inputs)...); };
    }
  };

}


#endif //OPENKALMAN_LINEARTRANSFORMATION_HPP
