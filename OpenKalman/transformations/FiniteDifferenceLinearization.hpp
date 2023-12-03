/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for FiniteDifferenceLinearization. 
 */

#ifndef OPENKALMAN_FINITEDIFFERENCELINEARIZATION_HPP
#define OPENKALMAN_FINITEDIFFERENCELINEARIZATION_HPP

#include <tuple>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


  /**
   * \brief A tests which calculates the first and second Taylor derivatives using finite differences.
   * \tparam Function The function to be linearized by finite differences.
   * \tparam InDelta The type of the input (and its delta).
   * \tparam PsDelta The type of the perturbations (and their deltas).
   */
#ifdef __cpp_concepts
  template<typename Function, transformation_input InDelta, transformation_input ... PsDelta> requires
    std::invocable<Function, InDelta, PsDelta...> and
    (not wrapped_mean<std::invoke_result_t<Function, InDelta, PsDelta...>>) and
    (not std::is_reference_v<InDelta>) and (((not std::is_reference_v<PsDelta>) and ...)) and
    ((sizeof...(PsDelta) == 0) or (equivalent_to<vector_space_descriptor_of_t<PsDelta, 0>,
      vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...))
#else
  template<typename Function, typename InDelta, typename ... PsDelta>
#endif
  struct FiniteDifferenceLinearization;


  namespace internal
  {
#ifdef __cpp_concepts
    template<typename Function, transformation_input InDelta, transformation_input ... PsDelta, std::size_t order>
      requires (order <= 2)
    struct is_linearized_function<FiniteDifferenceLinearization<Function, InDelta, PsDelta...>, order>
      : std::true_type {};
#else
  template<typename Function, typename InDelta, typename ... PsDelta, std::size_t order>
    struct is_linearized_function<FiniteDifferenceLinearization<Function, InDelta, PsDelta...>, order, std::enable_if_t<
      order <= 2>> : std::true_type {};
#endif
  }


#ifdef __cpp_concepts
  template<typename Function, transformation_input InDelta, transformation_input ... PsDelta> requires
    std::invocable<Function, InDelta, PsDelta...> and
    (not wrapped_mean<std::invoke_result_t<Function, InDelta, PsDelta...>>) and
    (not std::is_reference_v<InDelta>) and (((not std::is_reference_v<PsDelta>) and ...)) and
    ((sizeof...(PsDelta) == 0) or (equivalent_to<vector_space_descriptor_of_t<PsDelta, 0>,
      vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...))
#else
  template<typename Function, typename InDelta, typename ... PsDelta>
#endif
  struct FiniteDifferenceLinearization
  {

  private:

#ifndef __cpp_concepts
    static_assert((transformation_input<InDelta> and ... and transformation_input<PsDelta>));
    static_assert(std::is_invocable_v<Function, InDelta, PsDelta...>);
    static_assert(not wrapped_mean<std::invoke_result_t<Function, InDelta, PsDelta...>>,
      "For finite difference linearization, the tests function cannot return a wrapped matrix.");
    static_assert(not std::is_reference_v<InDelta>);
    static_assert(((not std::is_reference_v<PsDelta>) and ...));
    static_assert((sizeof...(PsDelta) == 0) or (equivalent_to<vector_space_descriptor_of_t<PsDelta, 0>,
      vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...));
#endif

    // Construct one Jacobian term.
    template<std::size_t term, typename...Inputs>
    auto jac_term(const std::tuple<Inputs...>& inputs) const
    {
      using Term = decltype(std::get<term>(inputs));
      using Scalar = scalar_type_of_t<Term>;
      constexpr auto width = index_dimension_of_v<Term, 0>;
      auto& col = std::get<term>(inputs);
      return apply_columnwise<width>([&](std::size_t i) {
        const Scalar h = std::get<term>(deltas)[i];
        const Scalar x = col[i];
        col[i] = x + h;
        const auto fp = Mean {std::apply(transformation, inputs)};
        col[i] = x - h;
        const auto fm = Mean {std::apply(transformation, inputs)};
        col[i] = x;
        return make_self_contained((fp - fm)/(2*h));
      });
    }


    // Collect Jacobian terms into a tuple.
#ifdef __cpp_concepts
    template<typename...Inputs, std::size_t...terms> requires (sizeof...(Inputs) == sizeof...(terms))
#else
    template<typename...Inputs, std::size_t...terms, std::enable_if_t<sizeof...(Inputs) == sizeof...(terms), int> = 0>
#endif
    auto jacobian_impl(const std::tuple<Inputs...>& inputs, std::index_sequence<terms...>) const
    {
      return std::tuple {jac_term<terms>(inputs) ...};
    }


    template<std::size_t term, std::size_t i, std::size_t j, typename...Inputs>
    auto h_j(const std::tuple<Inputs...>& inputs) const
    {
      using Scalar = scalar_type_of_t<decltype(std::get<term>(inputs))>;
      const Scalar hi = std::get<term>(deltas)[i];
      auto& col = std::get<term>(inputs);
      const Scalar xi = col[i];

      if constexpr (i == j)
      {
        const auto f0 = Mean {std::apply(transformation, inputs)};
        col[i] = xi + hi;
        const auto fp = Mean {std::apply(transformation, inputs)};
        col[i] = xi - hi;
        const auto fm = Mean {std::apply(transformation, inputs)};
        col[i] = xi;

        // Use two separate subtractions to ensure proper wrapping:
        auto ret {make_self_contained(((fp - f0) - (f0 - fm)) / (hi * hi))};
        return ret;
      }
      else
      {
        const Scalar hj = std::get<term>(deltas)[j];
        const Scalar xj = col[j];
        col[i] = xi + hi;
        col[j] = xj + hj;
        const auto fpp = Mean {std::apply(transformation, inputs)};
        col[j] = xj - hj;
        const auto fpm = Mean {std::apply(transformation, inputs)};
        col[i] = xi - hi;
        const auto fmm = Mean {std::apply(transformation, inputs)};
        col[j] = xj + hj;
        const auto fmp = Mean {std::apply(transformation, inputs)};
        col[i] = xi;
        col[j] = xj;
        auto ret {make_self_contained(((fpp - fpm) - (fmp - fmm)) / (4 * hi * hj))};
        return ret;
      }
    };


    template<std::size_t term, std::size_t i, typename...Inputs, std::size_t...js>
    auto h_i(const std::tuple<Inputs...>& inputs, std::index_sequence<js...>) const
    {
      using A = equivalent_self_contained_t<decltype(h_j<term, 0, 0>(inputs))>;
      return std::array<A, sizeof...(js)> {h_j<term, i, js>(inputs)...};
    }


    template<std::size_t term, typename...Inputs, std::size_t...is>
    auto h_k(const std::tuple<Inputs...>& inputs, std::index_sequence<is...>) const
    {
      constexpr auto j_size = index_dimension_of_v<decltype(std::get<term>(inputs)), 0>;
      using A = decltype(h_i<term, 0>(std::move(inputs), std::make_index_sequence<j_size> {}));
      return std::array<A, sizeof...(is)> {h_i<term, is>(inputs, std::make_index_sequence<j_size> {})...};
    }


    // For each hessian term, construct an array of Hessian matrices, one for each output dimension ks.
    template<std::size_t term, typename...Inputs, std::size_t...ks>
    auto h_term(const std::tuple<Inputs...>& inputs, std::index_sequence<ks...>) const
    {
      using Term = decltype(std::get<term>(inputs));
      const auto t = h_k<term>(inputs, std::make_index_sequence<index_dimension_of_v<Term, 0>> {});
      using C = vector_space_descriptor_of_t<Term, 0>;
      using V = Matrix<C, C, dense_writable_matrix_t<Term, Layout::none, scalar_type_of_t<Term>, Dimensions<index_dimension_of_v<Term, 0>>, Dimensions<index_dimension_of_v<Term, 0>>>>;
      return std::array<V, sizeof...(ks)> {
        apply_coefficientwise<V>([&t](std::size_t i, std::size_t j) { return t[i][j][ks]; })...};
    }


    // Construct a tuple of Hessian terms, one array for each input/perturbation term.
    template<typename...Inputs, std::size_t...terms>
    auto hessian_impl(const std::tuple<Inputs...>& inputs, std::index_sequence<terms...>) const
    {
      static_assert(sizeof...(Inputs) == sizeof...(terms));
      constexpr auto k_size = index_dimension_of_v<std::invoke_result_t<Function, Inputs...>, 0>;
      return std::tuple {h_term<terms>(inputs, std::make_index_sequence<k_size> {})...};
    }

  public:

    /**
     * \brief Constructor
     */
#ifdef __cpp_concepts
    template<typename T, transformation_input In, transformation_input ... Ps>
#else
    template<typename T, typename In, typename ... Ps, std::enable_if_t<
      (transformation_input<In> and ... and transformation_input<Ps>), int> = 0>
#endif
    FiniteDifferenceLinearization(T&& trans, In&& in_delta, Ps&& ... ps_delta)
      : transformation(std::forward<T>(trans)), deltas(std::forward<In>(in_delta), std::forward<Ps>(ps_delta)...) {}


    /// Applies the tests.
#ifdef __cpp_concepts
    template<transformation_input<vector_space_descriptor_of_t<InDelta, 0>> In, perturbation ... Perturbations>
      requires (sizeof...(Perturbations) <= sizeof...(PsDelta)) and (sizeof...(Perturbations) == 0 or
        (equivalent_to<typename oin::PerturbationTraits<Perturbations>::RowCoefficients,
          vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...))
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      transformation_input<In, vector_space_descriptor_of_t<InDelta, 0>> and
      (perturbation<Perturbations> and ...) and (sizeof...(Perturbations) <= sizeof...(PsDelta)) and
      (sizeof...(Perturbations) == 0 or
        (equivalent_to<typename oin::PerturbationTraits<Perturbations>::RowCoefficients,
          vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...)), int> = 0>
#endif
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      return transformation(std::forward<In>(in), std::forward<Perturbations>(ps)...);
    }


    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input<vector_space_descriptor_of_t<InDelta, 0>> In, perturbation ... Perturbations>
    requires (sizeof...(Perturbations) <= sizeof...(PsDelta)) and (sizeof...(Perturbations) == 0 or
      (equivalent_to<typename oin::PerturbationTraits<Perturbations>::RowCoefficients,
        vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...))
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      transformation_input<In, vector_space_descriptor_of_t<InDelta, 0>> and
      (perturbation<Perturbations> and ...) and (sizeof...(Perturbations) <= sizeof...(PsDelta)) and
      (sizeof...(Perturbations) == 0 or
        (equivalent_to<typename oin::PerturbationTraits<Perturbations>::RowCoefficients,
          vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...)), int> = 0>
#endif
    auto jacobian(In&& in, Perturbations&&...ps) const
    {
      return jacobian_impl(
        std::forward_as_tuple(make_dense_writable_matrix_from(std::forward<In>(in)),
          make_dense_writable_matrix_from(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<1 + sizeof...(ps)> {});
    }


    /// Returns a tuple of Hessian matrices for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input<vector_space_descriptor_of_t<InDelta, 0>> In, perturbation ... Perturbations>
    requires (sizeof...(Perturbations) <= sizeof...(PsDelta)) and (sizeof...(Perturbations) == 0 or
      (equivalent_to<typename oin::PerturbationTraits<Perturbations>::RowCoefficients,
        vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...))
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      transformation_input<In, vector_space_descriptor_of_t<InDelta, 0>> and
      (perturbation<Perturbations> and ...) and (sizeof...(Perturbations) <= sizeof...(PsDelta)) and
      (sizeof...(Perturbations) == 0 or
        (equivalent_to<typename oin::PerturbationTraits<Perturbations>::RowCoefficients,
          vector_space_descriptor_of_t<std::tuple_element_t<0, std::tuple<PsDelta...>>, 0>> and ...)), int> = 0>
#endif
    auto hessian(In&& in, Perturbations&&...ps) const
    {
      return hessian_impl(
        std::forward_as_tuple(make_dense_writable_matrix_from(std::forward<In>(in)),
          make_dense_writable_matrix_from(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<1 + sizeof...(ps)> {});
    }

  private:

    Function transformation;

    const std::tuple<const InDelta, const PsDelta...> deltas;

  };


  /*
   * Deduction guide
   */

#ifdef __cpp_concepts
  template<typename Function, transformation_input InDelta, transformation_input ... PsDelta> requires
  std::invocable<Function, InDelta, PsDelta...> and
    (not wrapped_mean<std::invoke_result_t<Function, InDelta, PsDelta...>>) and
  (not std::is_reference_v<InDelta>) and (((not std::is_reference_v<PsDelta>) and ...))
#else
  template<typename Function, typename InDelta, typename ... PsDelta, std::enable_if_t<
    (transformation_input<InDelta> and ... and transformation_input<PsDelta>) and
    std::is_invocable_v<Function, InDelta, PsDelta...> and
    (not wrapped_mean<std::invoke_result_t<Function, InDelta, PsDelta...>>) and
  (not std::is_reference_v<InDelta>) and (((not std::is_reference_v<PsDelta>) and ...)), int> = 0>
#endif
  FiniteDifferenceLinearization(Function&&, InDelta&&, PsDelta&&...)
    -> FiniteDifferenceLinearization<std::decay_t<Function>, std::decay_t<InDelta>, std::decay_t<PsDelta>...>;

}


#endif //OPENKALMAN_FINITEDIFFERENCELINEARIZATION_HPP
