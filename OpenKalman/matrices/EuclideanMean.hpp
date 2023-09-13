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
  template<fixed_index_descriptor RowCoefficients, typed_matrix_nestable NestedMatrix> requires
    (euclidean_dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct EuclideanMean : oin::TypedMatrixBase<EuclideanMean<RowCoefficients, NestedMatrix>, NestedMatrix,
    RowCoefficients, Dimensions<column_dimension_of_v<NestedMatrix>>>
  {

#ifndef __cpp_concepts
    static_assert(fixed_index_descriptor<RowCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(euclidean_dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  protected:

    using ColumnCoefficients = Dimensions<column_dimension_of_v<NestedMatrix>>;

  private:

    using Base = oin::TypedMatrixBase<EuclideanMean, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible Euclidean-transformed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      (equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients>) and
      (equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients>) and
      //requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } // \todo doesn't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      (equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients>) and
      (equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients>) and
      (equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients>) and
      requires(Arg&& arg) { NestedMatrix {to_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients>) and
      (equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients>) and
      std::is_constructible_v<NestedMatrix,
        decltype(to_euclidean<RowCoefficients>(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) noexcept
      : Base {to_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable" object.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (row_dimension_of_v<Arg> == row_dimension_of_v<NestedMatrix>) and
      (column_dimension_of_v<Arg> == column_dimension_of_v<NestedMatrix>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (row_dimension_of<Arg>::value == row_dimension_of<NestedMatrix>::value) and
      (column_dimension_of<Arg>::value == column_dimension_of<NestedMatrix>::value) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
     * \details This is operable where no tests to Euclidean space is required.
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, EuclideanMean>) and
      (euclidean_transformed<Arg> or euclidean_index_descriptor<RowCoefficients>) and
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      has_untyped_index<Arg, 1> and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not std::is_base_of_v<EuclideanMean, std::decay_t<Arg>>) and
      (euclidean_transformed<Arg> or euclidean_index_descriptor<RowCoefficients>) and
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      has_untyped_index<Arg, 1> and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_matrix(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix".
     * \details This is operable where a tests to Euclidean space is required.
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, EuclideanMean>) and
      (not euclidean_transformed<Arg> and fixed_index_descriptor<RowCoefficients>) and
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      has_untyped_index<Arg, 1> and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not std::is_base_of_v<EuclideanMean, std::decay_t<Arg>>) and
      (not euclidean_transformed<Arg> and fixed_index_descriptor<RowCoefficients>) and
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      has_untyped_index<Arg, 1> and modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(to_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(other))));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
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
      this->nested_matrix() += other.nested_matrix();
      return *this;
    }


    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients> and
      (euclidean_index_descriptor<RowCoefficients> or euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients> and
      (euclidean_index_descriptor<RowCoefficients> or euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      this->nested_matrix() += nested_matrix(std::forward<Arg>(other));
      return *this;
    }


    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires euclidean_index_descriptor<RowCoefficients> and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and euclidean_index_descriptor<RowCoefficients> and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      apply_columnwise([&arg](auto& col){ col += arg().nested_matrix(); }, this->nested_matrix());
      return *this;
    }


    /// Decrement from another EuclideanMean.
    auto& operator-=(const EuclideanMean& other)
    {
      this->nested_matrix() -= other.nested_matrix();
      return *this;
    }


    /// Decrement from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients> and
      (euclidean_index_descriptor<RowCoefficients> or euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<row_index_descriptor_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_index_descriptor_of_t<Arg>, ColumnCoefficients> and
      (euclidean_index_descriptor<RowCoefficients> or euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      this->nested_matrix() -= nested_matrix(std::forward<Arg>(other));
      return *this;
    }


    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires euclidean_index_descriptor<RowCoefficients> and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and euclidean_index_descriptor<RowCoefficients> and
      (equivalent_to<typename DistributionTraits<Arg>::TypedIndex, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      apply_columnwise([&arg](auto& col){ col -= arg().nested_matrix(); }, this->nested_matrix());
      return *this;
    }

  protected:

    template<typename C = RowCoefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
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
    (euclidean_dimension_size_of_v<row_index_descriptor_of_t<V>> == row_dimension_of_v<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V> and has_untyped_index<V, 1> and
euclidean_dimension_size_of_v<row_index_descriptor_of_t<V>> == row_dimension_of_v<V>, int> = 0>
#endif
  EuclideanMean(V&&)
    -> EuclideanMean<row_index_descriptor_of_t<V>, std::remove_reference_t<
      decltype(to_euclidean<row_index_descriptor_of_t<V>>(nested_matrix(std::declval<V&&>())))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and euclidean_transformed<V>, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<row_index_descriptor_of_t<V>, nested_matrix_of_t<V>>;


  /// Deduce template parameters from a typed_matrix_nestable, assuming axis-only coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  explicit EuclideanMean(V&&) -> EuclideanMean<Dimensions<row_dimension_of_v<V>>, passable_t<V>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a EuclideanMean from a typed_matrix_nestable, specifying the row coefficients.
   * \tparam TypedIndex The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typed_matrix_nestable M> requires
    (euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of_v<M>)
#else
  template<typename TypedIndex, typename M, std::enable_if_t<fixed_index_descriptor<TypedIndex> and
    typed_matrix_nestable<M> and (euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of<M>::value), int> = 0>
#endif
  auto make_euclidean_mean(M&& arg) noexcept
  {
    return EuclideanMean<TypedIndex, passable_t<M>>(std::forward<M>(arg));
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
  auto make_euclidean_mean(M&& m) noexcept
  {
    using Coeffs = Dimensions<row_dimension_of_v<M>>;
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
  inline auto make_euclidean_mean(Arg&& arg) noexcept
  {
    using C = row_index_descriptor_of_t<Arg>;
    if constexpr(euclidean_transformed<Arg>)
      return make_euclidean_mean<C>(nested_matrix(std::forward<Arg>(arg)));
    else
      return make_euclidean_mean<C>(nested_matrix(to_euclidean<C>(std::forward<Arg>(arg))));
  }


  /**
   * \overload
   * \brief Make a default, self-contained EuclideanMean.
   * \tparam TypedIndex The coefficient types corresponding to the rows.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typed_matrix_nestable M> requires
    (euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of_v<M>)
#else
  template<typename TypedIndex, typename M, std::enable_if_t<fixed_index_descriptor<TypedIndex> and
    typed_matrix_nestable<M> and (euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of<M>::value), int> = 0>
#endif
  auto make_euclidean_mean()
  {
    return EuclideanMean<TypedIndex, dense_writable_matrix_t<M, scalar_type_of_t<M>>>();
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
    return make_euclidean_mean<Dimensions<row_dimension_of_v<M>>, M>();
  }


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Coeffs, typename NestedMatrix>
    struct IndexibleObjectTraits<EuclideanMean<Coeffs, NestedMatrix>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<NestedMatrix>;

      using index_type = index_type_of_t<NestedMatrix>;

      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg, typename N>
      static constexpr auto get_index_descriptor(Arg&& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          if constexpr (static_index_value_of_v<N> == 0) return std::forward<Arg>(arg).my_dimension;
          else return OpenKalman::get_index_descriptor(nested_matrix(std::forward<Arg>(arg)), n);
        }
        else
        {
          using Scalar = scalar_type_of<Arg>;
          if (n == 0) return DynamicTypedIndex<Scalar> {std::forward<Arg>(arg).my_dimension};
          else return DynamicTypedIndex<Scalar> {OpenKalman::get_index_descriptor(nested_matrix(std::forward<Arg>(arg)), n)};
        }
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = one_by_one_matrix<NestedMatrix, b>;

      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<NestedMatrix>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).nested_matrix();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto n = make_self_contained(std::forward<Arg>(arg).nested_matrix());
        return EuclideanMean<Coeffs, decltype(n)> {std::move(n)};
      }

      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (euclidean_index_descriptor<Coeffs>)
          return constant_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }

      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        if constexpr (euclidean_index_descriptor<Coeffs>)
          return constant_diagonal_coefficient {arg.nestedExpression()};
        else
          return std::monostate {};
      }

      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = euclidean_index_descriptor<Coeffs> and triangular_matrix<NestedMatrix, t, b>;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_hermitian = euclidean_index_descriptor<Coeffs> and hermitian_matrix<NestedMatrix>;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires element_gettable<nested_matrix_of_t<Arg&&>, sizeof...(I)>
  #else
      template<typename Arg, typename...I, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg&&>::type, sizeof...(I)>, int> = 0>
  #endif
      static constexpr decltype(auto) get(Arg&& arg, I...i)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i...);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename I, typename...Is> requires element_settable<nested_matrix_of_t<Arg&>, 1 + sizeof...(Is)>
  #else
      template<typename Arg, typename I, typename...Is, std::enable_if_t<element_settable<typename nested_matrix_of<Arg&>::type, 1 + sizeof...(Is)>, int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
      {
        set_element(nested_matrix(arg), s, i, is...);
      }


      static constexpr bool is_writable = LibraryRoutines<std::decay_t<NestedMatrix>>::is_writable;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires directly_accessible<nested_matrix_of_t<Arg&&>> and euclidean_index_descriptor<Coeffs>
#else
      template<typename Arg, std::enable_if_t<directly_accessible<typename nested_matrix_of<Arg&&>::type> and euclidean_index_descriptor<Coeffs>, int> = 0>
#endif
      static constexpr auto*
      data(Arg& arg) { return internal::raw_data(nested_matrix(arg)); }


      static constexpr Layout layout = euclidean_index_descriptor<Coeffs> ? layout_of_v<NestedMatrix> : Layout::none;

    };

  } // namespace interface


} // namespace OpenKalman::internal


#endif //OPENKALMAN_EUCLIDEANMEAN_HPP
