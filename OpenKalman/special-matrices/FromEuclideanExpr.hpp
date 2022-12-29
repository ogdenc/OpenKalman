/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief FromEuclideanExpr and related definitions.
 */

#ifndef OPENKALMAN_FROMEUCLIDEANEXPR_HPP
#define OPENKALMAN_FROMEUCLIDEANEXPR_HPP

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typename NestedMatrix>
  requires (dynamic_index_descriptor<TypedIndex> == dynamic_rows<NestedMatrix>) and
    (not fixed_index_descriptor<TypedIndex> or euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not dynamic_index_descriptor<TypedIndex> or
      std::same_as<typename TypedIndex::Scalar, scalar_type_of_t<NestedMatrix>>)
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct FromEuclideanExpr : OpenKalman::internal::TypedMatrixBase<
    FromEuclideanExpr<TypedIndex, NestedMatrix>, NestedMatrix, TypedIndex>
  {

#ifndef __cpp_concepts
    static_assert(fixed_index_descriptor<TypedIndex>);
    static_assert(dynamic_index_descriptor<TypedIndex> == dynamic_rows<NestedMatrix>);
    static_assert(not fixed_index_descriptor<TypedIndex> or euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>;

  private:

    static constexpr auto columns = column_dimension_of_v<NestedMatrix>; ///< Number of columns.

    using Base = OpenKalman::internal::TypedMatrixBase<FromEuclideanExpr, NestedMatrix, TypedIndex>;

  public:

    using Base::Base;

    /**
     * Convert from a compatible from-euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    FromEuclideanExpr(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /**
     * Construct from a compatible to-euclidean expression.
     */
#ifdef __cpp_concepts
    template<to_euclidean_expr Arg> requires equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<to_euclidean_expr<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& other) noexcept : Base {std::forward<Arg>(other)} {}


    /**
     * Construct from compatible matrix object.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from compatible matrix object and an \ref index_descriptor.
#ifdef __cpp_concepts
    template<indexible Arg, index_descriptor C> requires (not euclidean_expr<Arg>) and
      std::constructible_from<NestedMatrix, Arg&&> and
      (dynamic_index_descriptor<C> or dynamic_index_descriptor<TypedIndex> or equivalent_to<C, TypedIndex>)
#else
    template<typename Arg, typename C, std::enable_if_t<indexible<Arg> and index_descriptor<C> and
      (not euclidean_expr<Arg>) and std::is_constructible_v<NestedMatrix, Arg&&> and
      (dynamic_index_descriptor<C> or dynamic_index_descriptor<TypedIndex> or equivalent_to<C, TypedIndex>), int> = 0>
#endif
    explicit FromEuclideanExpr(Arg&& arg, const TypedIndex& c) noexcept : Base {std::forward<Arg>(arg), c} {}


#ifndef __cpp_concepts
    /**
     * /brief Construct from a list of coefficients.
     * /note If c++ concepts are available, this functionality is inherited from the base class.
     */
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      sizeof...(Args) == columns *
        (to_euclidean_expr<NestedMatrix> ? dimension_size_of_v<TypedIndex> : euclidean_dimension_size_of_v<TypedIndex>), int> = 0>
    FromEuclideanExpr(Args ... args) : Base {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)} {}
#endif


    /**
     * Assign from a compatible from-Euclidean expression.
     */
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, FromEuclideanExpr>) and
      (equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>) and
      (column_dimension_of_v<Arg> == columns) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and
      (not std::is_base_of_v<FromEuclideanExpr, std::decay_t<Arg>>) and
      (equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>) and
      (column_dimension_of<Arg>::value == columns) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * Assign from a general Eigen matrix.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and
      (row_dimension_of_v<Arg> == dimension_size_of_v<TypedIndex>) and (column_dimension_of_v<Arg> == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<TypedIndex>(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (row_dimension_of<Arg>::value == dimension_size_of_v<TypedIndex>) and (column_dimension_of<Arg>::value == columns) and
      modifiable<NestedMatrix, decltype(to_euclidean<TypedIndex>(std::declval<Arg&&>()))>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        this->nested_matrix() = to_euclidean<TypedIndex>(std::forward<Arg>(arg));
      }
      return *this;
    }

  private:

    template<typename Arg>
    static auto to_euclidean_noalias(Arg&& arg)
    {
      if constexpr (euclidean_dimension_size_of_v<TypedIndex> > dimension_size_of_v<TypedIndex>)
        return make_dense_writable_matrix_from(to_euclidean<TypedIndex>(std::forward<Arg>(arg))); //< Prevent aliasing
      else
        return to_euclidean<TypedIndex>(make_self_contained<Arg>(std::forward<Arg>(arg)));
    }

  public:

    /// Increment from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (column_dimension_of_v<Arg> == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (column_dimension_of<Arg>::value == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Increment from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and (column_dimension_of_v<Arg> == columns) and
      (row_dimension_of_v<Arg> == dimension_size_of_v<TypedIndex>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (column_dimension_of<Arg>::value == columns) and
      (row_dimension_of<Arg>::value == dimension_size_of_v<TypedIndex>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this + arg);
      return *this;
    }


    /// Decrement from another \ref from_euclidean_expr.
#ifdef __cpp_concepts
    template<from_euclidean_expr Arg> requires (column_dimension_of_v<Arg> == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<from_euclidean_expr<Arg> and (column_dimension_of<Arg>::value == columns) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /// Decrement from another \ref matrix.
#ifdef __cpp_concepts
    template<indexible Arg> requires (not euclidean_expr<Arg>) and (column_dimension_of_v<Arg> == columns) and
      (row_dimension_of_v<Arg> == dimension_size_of_v<TypedIndex>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and (not euclidean_expr<Arg>) and
      (column_dimension_of<Arg>::value == columns) and
      (row_dimension_of<Arg>::value == dimension_size_of_v<TypedIndex>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      this->nested_matrix() = to_euclidean_noalias(*this - arg);
      return *this;
    }


    /**
     * Multiply by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S scale)
    {
      this->nested_matrix() = to_euclidean_noalias(*this * scale);
      return *this;
    }


    /**
     * Divide by a scale factor.
     * \param scale The scale factor.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S scale)
    {
      this->nested_matrix() = to_euclidean_noalias(*this / scale);
      return *this;
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<indexible Arg, index_descriptor C>
#else
  template<typename Arg, typename C, std::enable_if_t<indexible<Arg> and index_descriptor<C>, int> = 0>
#endif
  FromEuclideanExpr(Arg&&, const C&) -> FromEuclideanExpr<C, passable_t<Arg>>;


} // OpenKalman


#endif //OPENKALMAN_FROMEUCLIDEANEXPR_HPP
