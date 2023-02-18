/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for Eigen3::EigenWrapper
 */

#ifndef OPENKALMAN_EIGENWRAPPER_HPP
#define OPENKALMAN_EIGENWRAPPER_HPP

namespace OpenKalman::Eigen3
{
  template<typename NestedMatrix>
  struct EigenWrapper : std::conditional_t<std::is_base_of_v<Eigen::ArrayBase<NestedMatrix>, NestedMatrix>,
      Eigen::ArrayBase<EigenWrapper<NestedMatrix>>, Eigen::MatrixBase<EigenWrapper<NestedMatrix>>>
  {
  private:

    using Base = std::conditional_t<std::is_base_of_v<Eigen::ArrayBase<NestedMatrix>, NestedMatrix>,
      Eigen::ArrayBase<EigenWrapper<NestedMatrix>>, Eigen::MatrixBase<EigenWrapper<NestedMatrix>>>;


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct PacketScalarImpl { using type = scalar_type_of<NestedMatrix>; };


#ifdef __cpp_concepts
    template<typename T> requires requires { typename T::PacketScalar; }
    struct PacketScalarImpl<T>
#else
    template<typename T>
    struct PacketScalarImpl<T, std::void_t<typename T::PacketScalar>>
#endif
      { using type = typename T::PacketScalar; };


  public:

#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<std::decay_t<Arg>, EigenWrapper>)
#else
    template<typename Arg, std::enable_if_t<not std::is_same_v<std::decay_t<Arg>, EigenWrapper>, int> = 0>
#endif
    explicit EigenWrapper(Arg&& arg) : wrapped_expression(std::forward<Arg>(arg)) {};


    /// \internal \note Eigen3 requires this.
    using Scalar = scalar_type_of_t<NestedMatrix>;

    using PacketScalar = typename PacketScalarImpl<Base>::type;


    /* \internal
     * \brief The underlying numeric type for composed scalar types.
     * \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar.
     */
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;


    /* \internal
     * \brief The return type for coefficient access.
     * \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is
     * either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access.
     */
    using CoeffReturnType = typename Base::CoeffReturnType;


    /**
     * \internal
     * \brief The type of *this that is used for nesting within other Eigen classes.
     * \note Eigen3 requires this as the type used when Derived is nested.
     */
    using Nested = EigenWrapper;

    using StorageKind [[maybe_unused]] = typename Eigen::internal::traits<NestedMatrix>::StorageKind;

    using StorageIndex [[maybe_unused]] = typename Eigen::internal::traits<NestedMatrix>::StorageIndex;


    enum CompileTimeTraits
    {
      RowsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<NestedMatrix>::RowsAtCompileTime,
      ColsAtCompileTime [[maybe_unused]] = Eigen::internal::traits<NestedMatrix>::ColsAtCompileTime,
      Flags [[maybe_unused]] = Eigen::internal::traits<NestedMatrix>::Flags,
      SizeAtCompileTime [[maybe_unused]] = Base::SizeAtCompileTime,
      MaxSizeAtCompileTime [[maybe_unused]] = Base::MaxSizeAtCompileTime,
      IsVectorAtCompileTime [[maybe_unused]] = Base::IsVectorAtCompileTime,
    };


    /**
     * \internal
     * \return The number of rows at runtime.
     * \note Eigen3 requires this, particularly in Eigen::EigenBase.
     */
    constexpr Eigen::Index rows() const
    {
      return get_index_dimension_of<0>(wrapped_expression);
    }


    /**
     * \internal
     * \return The number of columns at runtime.
     * \note Eigen3 requires this, particularly in Eigen::EigenBase.
     */
    constexpr Eigen::Index cols() const
    {
      return get_index_dimension_of<1>(wrapped_expression);
    }

    /**
     * \brief Synonym for zero().
     * \note Overrides Eigen::DenseBase<Derived>::Zero.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use make_zero_matrix_like() instead.")]]
    constexpr auto Zero()
    {
      static_assert(not has_dynamic_dimensions<NestedMatrix>);
      return make_zero_matrix_like(wrapped_expression);
    }


    /**
     * \brief Synonym for zero().
     * \note Overrides Eigen::DenseBase<Derived>::Zero.
     * \return A matrix, of the same size and shape, containing only zero coefficients.
     */
    [[deprecated("Use make_zero_matrix_like() instead.")]]
    static constexpr auto Zero(const Eigen::Index r, const Eigen::Index c)
    {
      return make_zero_matrix_like<NestedMatrix>(Dimensions{static_cast<std::size_t>(r)}, Dimensions{static_cast<std::size_t>(c)});
    }


    /**
     * \brief Synonym for identity().
     * \note Overrides Eigen::DenseBase<Derived>::Identity.
     * \return An identity matrix with the same or identified number of rows and columns.
     */
    [[deprecated("Use make_identity_matrix_like() instead.")]]
    constexpr auto Identity()
    {
      if constexpr(square_matrix<NestedMatrix>)
        return make_identity_matrix_like(wrapped_expression);
      else
        return NestedMatrix::Identity();
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto operator<<(const S& s)
    {
      return wrapped_expression.operator<<(s);
    }


#ifdef __cpp_concepts
    template<indexible Other>
#else
    template<typename Other, std::enable_if_t<indexible<Other>, int> = 0>
#endif
    auto operator<<(const Other& other)
    {
      return wrapped_expression.operator<<(other);
    }


    NestedMatrix wrapped_expression;

  };

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGENWRAPPER_HPP