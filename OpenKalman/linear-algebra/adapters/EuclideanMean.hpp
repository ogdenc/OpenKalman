/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EUCLIDEANMEAN_HPP
#define OPENKALMAN_EUCLIDEANMEAN_HPP

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // --------------- //
  //  EuclideanMean  //
  // --------------- //

#ifdef __cpp_concepts
  template<fixed_pattern RowCoefficients, typed_matrix_nestable NestedMatrix> requires
    (coordinates::stat_dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct EuclideanMean : oin::TypedMatrixBase<EuclideanMean<RowCoefficients, NestedMatrix>, NestedMatrix,
    RowCoefficients, Dimensions<index_dimension_of_v<NestedMatrix, 1>>>
  {

#ifndef __cpp_concepts
    static_assert(fixed_pattern<RowCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(coordinates::stat_dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  protected:

    using ColumnCoefficients = Dimensions<index_dimension_of_v<NestedMatrix, 1>>;

  private:

    using Base = oin::TypedMatrixBase<EuclideanMean, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible Euclidean-transformed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      (compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>) and
      (compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>) and
      //requires(Arg&& arg) { NestedMatrix {nested_object(std::forward<Arg>(arg))}; } // \todo doesn't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      (compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>) and
      (compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_object(std::declval<Arg&&>()))>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) : Base {nested_object(std::forward<Arg>(arg))} {}


    /// Construct from a compatible non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>) and
      (compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>) and
      requires(Arg&& arg) { NestedMatrix {to_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>) and
      (compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>) and
      std::is_constructible_v<NestedMatrix,
        decltype(to_euclidean<RowCoefficients>(nested_object(std::declval<Arg&&>())))>, int> = 0>
#endif
    EuclideanMean(Arg&& arg)
      : Base {to_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(arg)))} {}


    /// Construct from compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable" object.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (index_dimension_of_v<Arg, 0> == index_dimension_of_v<NestedMatrix, 0>) and
      (index_dimension_of_v<Arg, 1> == index_dimension_of_v<NestedMatrix, 1>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (index_dimension_of<Arg, 0>::value == index_dimension_of<NestedMatrix, 0>::value) and
      (index_dimension_of<Arg, 1>::value == index_dimension_of<NestedMatrix, 1>::value) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) : Base {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
     * \details This is operable where no tests to Euclidean space is required.
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, EuclideanMean>) and
      (euclidean_transformed<Arg> or coordinates::euclidean_pattern<RowCoefficients>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      has_untyped_index<Arg, 1> and std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not std::is_base_of_v<EuclideanMean, std::decay_t<Arg>>) and
      (euclidean_transformed<Arg> or coordinates::euclidean_pattern<RowCoefficients>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      has_untyped_index<Arg, 1> and std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_object(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
     * \details This is operable where a tests to Euclidean space is required.
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, EuclideanMean>) and
      (not euclidean_transformed<Arg> and fixed_pattern<RowCoefficients>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      has_untyped_index<Arg, 1> and std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not std::is_base_of_v<EuclideanMean, std::decay_t<Arg>>) and
      (not euclidean_transformed<Arg> and fixed_pattern<RowCoefficients>) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      has_untyped_index<Arg, 1> and std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(to_euclidean<RowCoefficients>(nested_object(std::forward<Arg>(other))));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, Arg&&>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * \brief Increment from another EuclideanMean.
     */
    auto& operator+=(const EuclideanMean& other)
    {
      this->nested_object() += other.nested_object();
      return *this;
    }


    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (coordinates::euclidean_pattern<RowCoefficients> or euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (coordinates::euclidean_pattern<RowCoefficients> or euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator+=(Arg&& other)
    {
      this->nested_object() += nested_object(std::forward<Arg>(other));
      return *this;
    }


    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires coordinates::euclidean_pattern<RowCoefficients> and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and coordinates::euclidean_pattern<RowCoefficients> and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      apply_columnwise([&arg](auto& col){ col += arg().nested_object(); }, this->nested_object());
      return *this;
    }


    /// Decrement from another EuclideanMean.
    auto& operator-=(const EuclideanMean& other)
    {
      this->nested_object() -= other.nested_object();
      return *this;
    }


    /// Decrement from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients> and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients> and
      (coordinates::euclidean_pattern<RowCoefficients> or euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, RowCoefficients>and
      compares_with<vector_space_descriptor_of_t<Arg, 1>, ColumnCoefficients>and
      (coordinates::euclidean_pattern<RowCoefficients> or euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator-=(Arg&& other)
    {
      this->nested_object() -= nested_object(std::forward<Arg>(other));
      return *this;
    }


    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires coordinates::euclidean_pattern<RowCoefficients> and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and coordinates::euclidean_pattern<RowCoefficients> and
      (compares_with<typename DistributionTraits<Arg>::StaticDescriptor, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      apply_columnwise([&arg](auto& col){ col -= arg().nested_object(); }, this->nested_object());
      return *this;
    }

  protected:

    template<typename C = RowCoefficients, typename Arg>
    static auto make(Arg&& arg)
    {
      return EuclideanMean<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>) and has_untyped_index<V, 1> and
    (coordinates::stat_dimension_of_v<vector_space_descriptor_of_t<V, 0>> == index_dimension_of_v<V, 0>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V> and has_untyped_index<V, 1> and
coordinates::stat_dimension_of_v<vector_space_descriptor_of_t<V, 0>> == index_dimension_of_v<V, 0>, int> = 0>
#endif
  EuclideanMean(V&&)
    -> EuclideanMean<vector_space_descriptor_of_t<V, 0>, std::remove_reference_t<
      decltype(to_euclidean<vector_space_descriptor_of_t<V, 0>>(nested_object(std::declval<V&&>())))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and euclidean_transformed<V>, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<vector_space_descriptor_of_t<V, 0>, nested_object_of_t<V>>;


  /// Deduce template parameters from a typed_matrix_nestable, assuming axis-only coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  explicit EuclideanMean(V&&) -> EuclideanMean<Dimensions<index_dimension_of_v<V, 0>>, passable_t<V>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a EuclideanMean from a typed_matrix_nestable, specifying the row coefficients.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, typed_matrix_nestable M> requires
    (coordinates::stat_dimension_of_v<StaticDescriptor> == index_dimension_of_v<M, 0>)
#else
  template<typename StaticDescriptor, typename M, std::enable_if_t<fixed_pattern<StaticDescriptor> and
    typed_matrix_nestable<M> and (coordinates::stat_dimension_of_v<StaticDescriptor> == index_dimension_of<M, 0>::value), int> = 0>
#endif
  auto make_euclidean_mean(M&& arg)
  {
    return EuclideanMean<StaticDescriptor, passable_t<M>>(std::forward<M>(arg));
  }


  /**
   * \overload
   * \brief Make a EuclideanMean from a typed_matrix_nestable object, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  auto make_euclidean_mean(M&& m)
  {
    using Coeffs = Dimensions<index_dimension_of_v<M, 0>>;
    return make_mean<Coeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a EuclideanMean from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1>, int> = 0>
#endif
  inline auto make_euclidean_mean(Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    if constexpr(euclidean_transformed<Arg>)
      return make_euclidean_mean<C>(nested_object(std::forward<Arg>(arg)));
    else
      return make_euclidean_mean<C>(nested_object(to_euclidean<C>(std::forward<Arg>(arg))));
  }


  /**
   * \overload
   * \brief Make a default, self-contained EuclideanMean.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, typed_matrix_nestable M> requires
    (coordinates::stat_dimension_of_v<StaticDescriptor> == index_dimension_of_v<M, 0>)
#else
  template<typename StaticDescriptor, typename M, std::enable_if_t<fixed_pattern<StaticDescriptor> and
    typed_matrix_nestable<M> and (coordinates::stat_dimension_of_v<StaticDescriptor> == index_dimension_of<M, 0>::value), int> = 0>
#endif
  auto make_euclidean_mean()
  {
    return EuclideanMean<StaticDescriptor, dense_writable_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained EuclideanMean with default Axis coefficients.
   * \tparam M a typed_matrix_nestable on which the new Euclidean mean is based.
   * It will be converted to a self_contained type if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  auto make_euclidean_mean()
  {
    return make_euclidean_mean<Dimensions<index_dimension_of_v<M, 0>>, M>();
  }


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Coeffs, typename NestedMatrix>
    struct indexible_object_traits<EuclideanMean<Coeffs, NestedMatrix>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return OpenKalman::count_indices(nested_object(arg)); }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
      {
        if constexpr (values::fixed<N>)
        {
          if constexpr (n == 0_uz) return arg.my_dimension;
          else return OpenKalman::get_vector_space_descriptor(nested_object(arg), n);
        }
        else if constexpr (uniform_static_vector_space_descriptor<NestedMatrix> and compares_with<Coeffs, uniform_static_vector_space_descriptor_component_of<NestedMatrix>>)
        {
          return arg.my_dimension;
        }
        else
        {
          if (n == 0) return DynamicDescriptor<scalar_type> {arg.my_dimension};
          else return DynamicDescriptor<scalar_type> {OpenKalman::get_vector_space_descriptor(nested_object(arg), n)};
        }
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).nested_object();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (coordinates::euclidean_pattern<Coeffs>)
          return constant_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (coordinates::euclidean_pattern<Coeffs>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<Applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedMatrix, b>;


      template<Applicability b>
      static constexpr bool is_square = OpenKalman::square_shaped<NestedMatrix, b>;


      template<TriangleType t>
      static constexpr bool is_triangular = coordinates::euclidean_pattern<Coeffs> and triangular_matrix<NestedMatrix, t>;


      static constexpr bool is_triangular_adapter = false;


      static constexpr bool is_hermitian = coordinates::euclidean_pattern<Coeffs> and hermitian_matrix<NestedMatrix>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<nested_object_of_t<Arg&&>, sizeof...(I)>
  #else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<typename nested_object_of<Arg&&>::type, sizeof...(I)>, int> = 0>
  #endif
      static constexpr decltype(auto)
      get(Arg&& arg, I...i)
      {
        return get_component(OpenKalman::nested_object(std::forward<Arg>(arg)), i...);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires writable_by_component<nested_object_of_t<Arg&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<writable_by_component<typename nested_object_of<Arg&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr void
      set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        set_component(OpenKalman::nested_object(arg), s, i, is...);
      }


      static constexpr bool is_writable = library_interface<std::decay_t<NestedMatrix>>::is_writable;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires raw_data_defined_for<NestedMatrix> and coordinates::euclidean_pattern<Coeffs>
#else
      template<typename Arg, std::enable_if_t<raw_data_defined_for<NestedMatrix> and coordinates::euclidean_pattern<Coeffs>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(OpenKalman::nested_object(arg)); }


      static constexpr Layout layout = coordinates::euclidean_pattern<Coeffs> ? layout_of_v<NestedMatrix> : Layout::none;

    };

  } // namespace interface


} // namespace OpenKalman::internal


#endif //OPENKALMAN_EUCLIDEANMEAN_HPP
